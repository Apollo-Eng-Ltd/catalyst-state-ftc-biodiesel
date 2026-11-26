"""Fault classification and isolation.

Models (all trained on the training split only; class weights balance the doubled normal class):
  res_par    causal GRU on whitened residuals (6) + log(1 + D) + estimated fault parameters (6,
             from the data-only estimator; CROSS-FITTED on the training split, i.e. each training batch's
             parameters come from an estimator that never saw it) + recipe. Labels every sample:
             the fault class once the fault is active, otherwise N (online isolation).
  res        same without estimated parameters.
  bilstm_raw bidirectional LSTM on the raw measurements + recipe, one label per batch (the base paper's
             approach; it sees the whole batch, so it is not usable online).
Batch-level decision for the causal models = prediction at the last sample (t = 120 min).
Metrics: confusion matrices, accuracy recomputed from the matrix and asserted equal to the direct count,
per-class recall, macro F1; online isolation delay for causal models; Jatropha charges should be F3.
Usage: python3 train_classifier.py
"""
import os, sys, json, time, datetime
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "estimator")); sys.path.insert(0, os.path.join(HERE, "..", "faults"))
from faults import CLASSES                                  # noqa: E402
from evaluate import net_predict                            # noqa: E402
from train_estimator import load, Net as ENet               # noqa: E402

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
DATA = os.path.join(HERE, "..", "data")
STRIDE, DT = 2, 0.5
SPLITS = ["train", "test_interp", "test_extrap", "test_jco"]
PARS = ["eta", "phi", "bias", "a_rel", "F_W", "k_loss"]


def est_params(tag, d):
    """Six fault-parameter estimates over time from a trained estimator."""
    ck = torch.load(os.path.join(HERE, "..", "estimator", f"est_{tag}.pt"), weights_only=False)
    net = ENet(ck["n_in"], ck["n_out"]).double(); net.load_state_dict(ck["state"]); net.eval()
    xin = np.concatenate([d["Xm"], np.repeat(d["st"][:, None, :], d["Xm"].shape[1], 1)], -1)
    with torch.no_grad():
        y = net(torch.tensor((xin - ck["mu_x"]) / ck["sd_x"])).numpy() * ck["sd_y"] + ck["mu_y"]
    return y[..., 13:19]


def features(split):
    d = load(split)
    R = np.load(f"{DATA}/{split}_R.npz")
    r, D, MU, SINV = R["R"][:, ::STRIDE], R["D"][:, ::STRIDE], R["MU"][::STRIDE], R["SINV"][::STRIDE]
    L = np.linalg.cholesky(SINV)                       # SINV = L L'
    w = np.einsum("tji,btj->bti", L, r - MU[None])     # whitened residuals
    if split == "train":                               # cross-fitted parameter estimates
        half = np.random.default_rng(7).permutation(len(d["meta"]))
        in0 = np.zeros(len(d["meta"]), bool); in0[half[: len(half) // 2]] = True
        p0, p1 = est_params("data_f0", d), est_params("data_f1", d)
        P = np.where(in0[:, None, None], p1, p0)       # batch in half 0 -> predicted by the fold-1 model
    else:
        P = est_params("data", d)
    X = np.load(f"{DATA}/{split}_X.npz", allow_pickle=True); xc = list(X["channels"])
    act = X["X"][:, ::STRIDE, xc.index("fault_active")]
    meta = d["meta"]
    lab_b = meta.cls.map({c: i for i, c in enumerate(CLASSES)}).values
    lab_t = np.where(act > 0.5, lab_b[:, None], 0)
    return dict(w=w, logD=np.log1p(D)[..., None], P=P, raw=d["Xm"], st=d["st"], lab_b=lab_b, lab_t=lab_t,
                meta=meta, t_f=meta.t_f.values)


class Causal(torch.nn.Module):
    def __init__(self, n_in, h=64):
        super().__init__()
        self.gru = torch.nn.GRU(n_in, h, num_layers=2, batch_first=True)
        self.out = torch.nn.Linear(h, len(CLASSES))

    def forward(self, x):
        return self.out(self.gru(x)[0])


class BiLSTM(torch.nn.Module):
    def __init__(self, n_in, h=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_in, h, num_layers=2, batch_first=True, bidirectional=True)
        self.out = torch.nn.Linear(2 * h, len(CLASSES))

    def forward(self, x):
        return self.out(self.lstm(x)[0].mean(1))


def inputs(F, kind):
    T = F["w"].shape[1]
    st = np.repeat(F["st"][:, None, :], T, 1)
    if kind == "res_par":
        return np.concatenate([F["w"], F["logD"], F["P"], st], -1)
    if kind == "res":
        return np.concatenate([F["w"], F["logD"], st], -1)
    return np.concatenate([F["raw"], st], -1)


def train(kind, Ftr, epochs=30):
    x = inputs(Ftr, kind)
    mu, sd = x.reshape(-1, x.shape[-1]).mean(0), x.reshape(-1, x.shape[-1]).std(0) + 1e-9
    X = torch.tensor((x - mu) / sd, dtype=torch.float32)
    counts = np.bincount(Ftr["lab_b"], minlength=len(CLASSES))
    wcls = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
    model = (BiLSTM if kind == "bilstm_raw" else Causal)(x.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    yb, yt = torch.tensor(Ftr["lab_b"]), torch.tensor(Ftr["lab_t"])
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(X)); tot = 0
        for k in range(0, len(perm), 32):
            bi = torch.tensor(perm[k:k + 32])
            out = model(X[bi])
            if kind == "bilstm_raw":
                loss = torch.nn.functional.cross_entropy(out, yb[bi], weight=wcls)
            else:
                loss = torch.nn.functional.cross_entropy(out.reshape(-1, len(CLASSES)), yt[bi].reshape(-1), weight=wcls)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        print(f"{kind} ep {ep:2d} loss {tot / len(X):.4f}", flush=True)
    return model, mu, sd


def predict(model, kind, F, mu, sd):
    X = torch.tensor((inputs(F, kind) - mu) / sd, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model(X).numpy()
    return out                                               # (B, C) for bilstm, (B, T, C) causal


def confusion(y, p, n=len(CLASSES)):
    M = np.zeros((n, n), int)
    for a, b in zip(y, p):
        M[a, b] += 1
    return M


def metrics(M, y, p):
    acc_m = np.trace(M) / M.sum()
    acc_d = float(np.mean(np.asarray(y) == np.asarray(p)))
    assert abs(acc_m - acc_d) < 1e-12, (acc_m, acc_d)          # reported accuracy must match its matrix
    rows = M.sum(1); cols = M.sum(0)
    rec = np.where(rows > 0, np.diag(M) / np.maximum(rows, 1), np.nan)
    prec = np.where(cols > 0, np.diag(M) / np.maximum(cols, 1), np.nan)
    f1 = np.where((rec + prec) > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-12), 0.0)
    present = rows > 0
    return dict(accuracy=round(float(acc_m), 4), macro_F1=round(float(np.nanmean(f1[present])), 4),
                recall={CLASSES[i]: round(float(rec[i]), 3) for i in range(len(CLASSES)) if present[i]})


def isolation_delay(out_t, F, hold=4):
    """First time after onset where the causal prediction equals the true class for `hold` samples (2 min)."""
    pred = out_t.argmax(-1); delays = []
    for b in range(len(pred)):
        c = F["lab_b"][b]
        if c == 0:
            continue
        i0 = int(np.ceil(F["t_f"][b] / DT)); run = 0; dly = np.nan
        for t in range(i0, pred.shape[1]):
            run = run + 1 if pred[b, t] == c else 0
            if run == hold:
                dly = (t - hold + 1) * DT - F["t_f"][b]; break
        delays.append((CLASSES[c], dly))
    df = pd.DataFrame(delays, columns=["cls", "delay"])
    return dict(isolated_frac=df.groupby("cls").delay.apply(lambda v: round(float(v.notna().mean()), 3)).to_dict(),
                median_delay_min=df.groupby("cls").delay.median().round(2).to_dict())


def main():
    t0 = time.time()
    F = {sp: features(sp) for sp in SPLITS}
    print("features ready", round(time.time() - t0), "s", flush=True)
    rep = dict(run_utc=datetime.datetime.now(datetime.UTC).isoformat(), classes=CLASSES, models={})
    for kind in ["res_par", "res", "bilstm_raw"]:
        model, mu, sd = train(kind, F["train"])
        torch.save(dict(state=model.state_dict(), mu=mu, sd=sd, kind=kind), os.path.join(HERE, f"clf_{kind}.pt"))
        rep["models"][kind] = {}
        for sp in SPLITS[1:]:
            out = predict(model, kind, F[sp], mu, sd)
            p = out.argmax(-1) if kind == "bilstm_raw" else out[:, -1, :].argmax(-1)
            y = F[sp]["lab_b"]
            M = confusion(y, p)
            pd.DataFrame(M, index=CLASSES, columns=CLASSES).to_csv(os.path.join(HERE, f"cm_{kind}_{sp}.csv"))
            r = metrics(M, y, p)
            if kind != "bilstm_raw" and sp != "test_jco":
                r["online"] = isolation_delay(out, F[sp])
            if sp == "test_jco":
                r["predicted_as"] = {CLASSES[i]: int(n) for i, n in enumerate(np.bincount(p, minlength=len(CLASSES))) if n}
            rep["models"][kind][sp] = r
            print(kind, sp, json.dumps({k: v for k, v in r.items() if k != "online"}), flush=True)
    rep["seconds"] = round(time.time() - t0)
    json.dump(rep, open(os.path.join(HERE, "classifier.run.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
