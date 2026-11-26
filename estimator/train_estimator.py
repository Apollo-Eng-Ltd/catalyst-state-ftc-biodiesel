"""Physics-informed causal estimator (and its data-only twin).

Causal 2-layer GRU (it sees measurements up to time t only) maps the eight online measurements and the
batch recipe to: scaled moles of 11 species, scaled volume, true reactor temperature, and six fault
parameters (eta, phi, sensor bias b, activity a_rel, water ingress F_W, methanol-loss rate k_loss).

Loss  L = L_data + w_meas L_meas + lam_phys L_phys + lam_IC L_IC
  L_data  standardized MSE on all labelled outputs (simulation ground truth)
  L_meas  (T_hat + b_hat - T_meas)^2 / sigma_T^2                         measurement consistency
  L_phys  species balances and reactor energy balance, central differences of the predicted
          trajectories against torch_plant (nominal parameters, network's own fault estimates)
  L_IC    predicted state at t = 0 equals the known charge (TG, methanol, base, zero products, V0)
Data at 0.5 min (every second sample). Validation: 10 % of training batches, by batch.
  v2 (17 Nov): L_meas also requires the predicted state to reproduce the measured apparent pH
  (observation equation of faults.py) and the load-cell mass (charge + integrated predicted
  methanol-loss and water-ingress flows).
Options: --labels f  data loss on a seeded fraction f of training batches only (others: physics + meas)
         --adapt <pool> --init <tag>  unlabelled fine-tuning on a new-feed pool (physics + meas only)
  v3 (--v3): the network also estimates four uncertain model parameters per batch (k_SE, and multipliers on
  k_H, c_th, UA), supervised in training and used inside the physics residual, so the physics constrains
  the model structure rather than its uncertain constants.
  --fold k (0/1): train on one seeded half of the training batches (cross-fitted predictions for the classifier).
Usage: python3 train_estimator.py <tag> <lam_phys> [epochs] [--labels f] [--adapt pool --init tag] [--nometa] [--v3]
"""
import os, sys, json, time, datetime
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import torch_plant as tp                                   # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "faults"))
from faults import PH_REF, C_B_REF, MW_FAME, RHO_FAME       # noqa: E402
from pilot import pilot_case                               # noqa: E402

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(1)
DATA = os.path.join(HERE, "..", "data")
STRIDE, DT = 2, 0.5
CL = "--cl" in sys.argv          # closed-loop excitation data (cl_* files) with known dosing inputs
PREFIX = "cl_" if CL else ""
MEAS = ["T", "T_J", "T_Jin", "T_cmd", "Q_J", "P_st", "pH", "m_ld"] + (["u_cat", "F_Ain"] if CL else [])
STATIC = ["cat_wt", "ratio", "AV", "water_wt", "Tsp_C", "T0_C"]
PARS = ["eta", "phi", "bias", "a_rel", "F_W", "k_loss"]
UNC = ["log_kSE", "kH_mult", "cth_mult", "UA_mult"]     # v3: uncertain model parameters, estimated per batch
SIG_T, SIG_PH, SIG_M = 0.05, 0.02, 0.2


def load(split):
    split = split if split.startswith(PREFIX) else PREFIX + split
    X = np.load(f"{DATA}/{split}_X.npz", allow_pickle=True); S = np.load(f"{DATA}/{split}_S.npz", allow_pickle=True)
    xc, sc = list(X["channels"]), list(S["channels"])
    meta = pd.read_csv(f"{DATA}/{split}_meta.csv")
    Xm = X["X"][:, ::STRIDE, :][:, :, [xc.index(c) for c in MEAS]].astype(np.float64)
    Sx = S["S"][:, ::STRIDE, :].astype(np.float64)
    nTG0 = Sx[:, 0, sc.index("TG")]; V0 = Sx[:, 0, sc.index("V")]
    z = np.stack([Sx[:, :, sc.index(s)] for s in tp.SP], -1) / nTG0[:, None, None]
    zV = Sx[:, :, sc.index("V")] / V0[:, None]
    Ttrue = Sx[:, :, sc.index("T")]
    nA = Sx[:, :, sc.index("A")]
    pars = np.stack([Sx[:, :, sc.index("eta")], Sx[:, :, sc.index("phi")], Sx[:, :, sc.index("bias")],
                     Sx[:, :, sc.index("a_rel")], Sx[:, :, sc.index("F_W")],
                     -Sx[:, :, sc.index("F_A")] / np.maximum(nA, 1e-9)], -1)
    rho_cp, UA = np.zeros(len(meta)), np.zeros(len(meta))
    for i, r in meta.iterrows():
        _, p = pilot_case(r.oil, ratio=r.ratio, cat_wt=r.cat_wt, AV=r.AV, water_wt=r.water_wt, T0_C=r.T0_C, group="veg")
        rho_cp[i], UA[i] = p.rho_cp, p.UA
    st = meta[STATIC].values.astype(np.float64)
    unc = np.stack([np.log(meta.mm_k_SE.values), meta.mm_k_H.values, meta.mm_c_th.values, meta.mm_UA.values], -1)
    return dict(Xm=Xm, st=st, z=z, zV=zV, T=Ttrue, pars=pars, nTG0=nTG0, V0=V0, rho_cp=rho_cp, UA=UA,
                TJ=Xm[:, :, MEAS.index("T_J")] + 273.15, Tmeas=Xm[:, :, MEAS.index("T")] + 273.15, meta=meta,
                pHmeas=Xm[:, :, MEAS.index("pH")], mmeas=Xm[:, :, MEAS.index("m_ld")], unc=unc)


class Net(torch.nn.Module):
    def __init__(self, n_in, n_out, h=96):
        super().__init__()
        self.gru = torch.nn.GRU(n_in, h, num_layers=2, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, n_out))

    def forward(self, x):
        return self.head(self.gru(x)[0])


V3 = "--v3" in sys.argv


def targets(d):
    base = [d["z"], d["zV"][..., None], d["T"][..., None], d["pars"]]
    if V3:
        base.append(np.repeat(d["unc"][:, None, :], d["z"].shape[1], 1))
    return np.concatenate(base, -1)


def main(tag, lam_phys, epochs, labels=1.0, adapt=None, init=None, use_meas2=True, opt_fold=None):
    t0 = time.time()
    tr = load("train")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(tr["meta"])); nval = len(idx) // 10
    va_i, tr_i = idx[:nval], idx[nval:]
    fold = int(opt_fold) if opt_fold is not None else None
    if fold is not None:                                 # cross-fitting: train on one half of the training batches
        half = np.random.default_rng(7).permutation(len(tr["meta"]))
        keep = set(half[: len(half) // 2] if fold == 0 else half[len(half) // 2:])
        tr_i = np.array([i for i in tr_i if i in keep]); va_i = np.array([i for i in va_i if i in keep])
    lab = np.zeros(len(tr["meta"]), bool)
    lab[rng.permutation(tr_i)[:int(round(labels * len(tr_i)))]] = True
    lab[va_i] = True                                     # validation always labelled (for monitoring only)
    # normalisation (training batches only)
    xin = np.concatenate([tr["Xm"], np.repeat(tr["st"][:, None, :], tr["Xm"].shape[1], 1)], -1)
    mu_x, sd_x = xin[tr_i].reshape(-1, xin.shape[-1]).mean(0), xin[tr_i].reshape(-1, xin.shape[-1]).std(0) + 1e-9
    Y = targets(tr)
    mu_y, sd_y = Y[tr_i].reshape(-1, Y.shape[-1]).mean(0), Y[tr_i].reshape(-1, Y.shape[-1]).std(0) + 1e-6
    nS = len(tp.SP)
    sc_z = torch.tensor(sd_y[:nS])                     # species scale for physics residual
    net = Net(xin.shape[-1], Y.shape[-1]).double()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    T_ = lambda a: torch.tensor(a, dtype=torch.float64)
    if adapt:                                            # unlabelled new-feed pool replaces the training set
        ck0 = torch.load(os.path.join(HERE, f"est_{init}.pt"), weights_only=False)
        mu_x, sd_x, mu_y, sd_y = ck0["mu_x"], ck0["sd_x"], ck0["mu_y"], ck0["sd_y"]
        sc_z = torch.tensor(sd_y[:nS])
        tr = load(adapt)
        xin = np.concatenate([tr["Xm"], np.repeat(tr["st"][:, None, :], tr["Xm"].shape[1], 1)], -1)
        Y = targets(tr)
        idx = rng.permutation(len(tr["meta"])); nval = max(1, len(idx) // 10); va_i, tr_i = idx[:nval], idx[nval:]
        lab = np.zeros(len(tr["meta"]), bool); lab[va_i] = True   # labels used ONLY to report validation error
        net.load_state_dict(ck0["state"])
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    if init and not adapt:                               # curriculum: start from a trained data-only model
        ck0 = torch.load(os.path.join(HERE, f"est_{init}.pt"), weights_only=False)
        mu_x, sd_x, mu_y, sd_y = ck0["mu_x"], ck0["sd_x"], ck0["mu_y"], ck0["sd_y"]
        sc_z = torch.tensor(sd_y[:nS])
        net.load_state_dict(ck0["state"])
        opt = torch.optim.Adam(net.parameters(), lr=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xn, Yn = T_((xin - mu_x) / sd_x), T_((Y - mu_y) / sd_y)
    LAB = torch.tensor(lab)
    aux = {k: T_(tr[k]) for k in ["nTG0", "V0", "rho_cp", "UA", "TJ", "Tmeas", "pHmeas", "mmeas"]}
    IC = np.zeros((len(tr["meta"]), nS + 1)); IC[:, :nS] = tr["z"][:, 0, :]; IC[:, nS] = 1.0
    IC = T_(IC); ic_mask = torch.ones(nS + 1, dtype=torch.float64); ic_mask[tp.IX["FFA"]] = 0; ic_mask[tp.IX["W"]] = 0
    MU, SD = T_(mu_y), T_(sd_y)

    def losses(bi):
        out = net(Xn[bi])
        lm = LAB[bi]
        L_data = ((out[lm] - Yn[bi][lm]) ** 2).mean() if lm.any() else torch.zeros((), dtype=torch.float64)
        y = out * SD + MU                                          # physical units
        z, zV, That = y[..., :nS], y[..., nS], y[..., nS + 1]
        eta, phi, b, a_rel, F_W, kl = (y[..., nS + 2 + j] for j in range(6))
        L_meas = (((That + b) - aux["Tmeas"][bi]) ** 2).mean() / SIG_T ** 2 / 100.0
        if use_meas2:
            n0 = aux["nTG0"][bi][:, None]
            V_bp = (torch.clamp(z[..., tp.IX["A"]], min=1e-6) * n0 * 32.042 / 0.791
                    + torch.clamp(z[..., tp.IX["E"]], min=0) * n0 * MW_FAME / RHO_FAME) / 1000.0
            base = torch.clamp(z[..., tp.IX["Cat"]] + z[..., tp.IX["OH"]], min=1e-6) * n0
            pHh = PH_REF + torch.log10(base / V_bp / C_B_REF)
            flow = (-torch.clamp(kl, min=0) * torch.clamp(z[..., tp.IX["A"]], min=0) * n0 * 32.042
                    + torch.clamp(F_W, min=0) * 18.015) / 1000.0                     # kg/min
            mh = aux["mmeas"][bi][:, :1] + torch.cat([torch.zeros_like(flow[:, :1]),
                                                       torch.cumsum(0.5 * (flow[:, 1:] + flow[:, :-1]) * DT, 1)], 1)
            L_meas = L_meas + ((pHh - aux["pHmeas"][bi]) ** 2).mean() / SIG_PH ** 2 / 100.0 \
                     + ((mh - aux["mmeas"][bi]) ** 2).mean() / SIG_M ** 2 / 100.0
        L_IC = ((((torch.cat([z[:, 0], zV[:, :1]], -1) - IC[bi]) / torch.cat([sc_z, SD[nS:nS + 1]])) ** 2) * ic_mask).mean()
        if lam_phys > 0:
            n0, V0 = aux["nTG0"][bi][:, None], aux["V0"][bi][:, None]
            if V3:
                u = y[..., nS + 8:nS + 12][:, 1:-1]
                kSE60 = torch.exp(torch.clamp(u[..., 0], np.log(1e-3), np.log(0.2)))
                kHm, cthm, UAm = (torch.clamp(u[..., j], 0.3, 3.0) for j in (1, 2, 3))
            else:
                kSE60 = kHm = cthm = None; UAm = 1.0
            dz, dzV, r = tp.dz_dt(z[:, 1:-1], zV[:, 1:-1], That[:, 1:-1], torch.clamp(a_rel[:, 1:-1], 0.05, 2),
                                  torch.clamp(eta[:, 1:-1], 0.05, 2), torch.clamp(F_W[:, 1:-1], min=0), torch.clamp(kl[:, 1:-1], min=0),
                                  n0, V0, kSE60, kHm, cthm)
            fd = (z[:, 2:] - z[:, :-2]) / (2 * DT)
            Rz = ((fd - dz) / sc_z * 10.0) ** 2                     # residual in units of species scale per 0.1 min
            dTm = tp.dT_dt(That[:, 1:-1], aux["TJ"][bi][:, 1:-1], r, zV[:, 1:-1], V0,
                           aux["rho_cp"][bi][:, None], aux["UA"][bi][:, None] * UAm)
            RT = (((That[:, 2:] - That[:, :-2]) / (2 * DT) - dTm) / 0.1) ** 2   # in units of 0.1 K/min
            L_phys = Rz[:, 3:].mean() + RT[:, 3:].mean()
        else:
            L_phys = torch.zeros((), dtype=torch.float64)
        return L_data, L_meas, L_phys, L_IC

    hist = []
    for ep in range(epochs):
        net.train(); perm = rng.permutation(tr_i); acc = np.zeros(4); nb = 0
        for k in range(0, len(perm), 32):
            bi = torch.tensor(perm[k:k + 32])
            Ld, Lm, Lp, Li = losses(bi)
            loss = (0.0 if adapt else 1.0) * Ld + Lm + lam_phys * Lp + 1.0 * Li
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            acc += [Ld.item(), Lm.item(), Lp.item(), Li.item()]; nb += 1
        sched.step()
        net.eval()
        with torch.no_grad():
            v = losses(torch.tensor(va_i))
        hist.append(dict(epoch=ep, train=(acc / nb).round(5).tolist(), val=[round(x.item(), 5) for x in v]))
        print(f"{tag} ep {ep:3d}  train {np.round(acc / nb, 4)}  val {[round(x.item(), 4) for x in v]}  {time.time() - t0:5.0f}s", flush=True)
    torch.save(dict(state=net.state_dict(), mu_x=mu_x, sd_x=sd_x, mu_y=mu_y, sd_y=sd_y, lam_phys=lam_phys, v3=V3,
                    n_in=xin.shape[-1], n_out=Y.shape[-1]), os.path.join(HERE, f"est_{tag}.pt"))
    json.dump(dict(tag=tag, lam_phys=lam_phys, epochs=epochs, labels=labels, adapt=adapt, init=init,
                   meas2=use_meas2, seconds=round(time.time() - t0), hist=hist,
                   run_utc=datetime.datetime.now(datetime.UTC).isoformat(), val_batches=int(nval)),
              open(os.path.join(HERE, f"est_{tag}.run.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    main(a[1], float(a[2]), int(a[3]) if len(a) > 3 and not a[3].startswith("--") else 40,
         labels=float(opt("--labels", 1.0)), adapt=opt("--adapt"), init=opt("--init"), use_meas2="--nometa" not in a,
         opt_fold=opt("--fold"))
