"""Batch dataset with held-out severities and onset windows.

Splits are made by whole batches, never within a batch, and by fault magnitude and onset:
  severity s in [0, 1] in five bands of 0.2
    train         bands 0, 1, 3      s in [0, 0.4) U [0.6, 0.8)
    test_interp   band 2             s in [0.4, 0.6)      (unseen magnitudes inside the training range)
    test_extrap   band 4             s in [0.8, 1.0]      (beyond the training range)
  onset t_f (min) for in-batch faults F2, E1-E4
    train         [5, 35) U [50, 70)
    test_interp   [35, 50)
    test_extrap   [70, 90]
  F1, F3 are in the charge (t_f = 0).
  test_jco: real Jatropha charges (fitted activity, no injected fault), labelled F3, never trained on.
Operating conditions by Latin hypercube (seeded): oil (six vegetable oils), K2CO3 3.5-5 wt%,
MeOH:oil 6-8, T0 50-55 C, T_sp 57-60 C, water 0.01-0.03 wt%, AV x 0.8-1.2 of the 2013 value.
Window chosen so that every fault-free batch reaches 96.5 % and stays below 64.7 C (checked on
120 LHS batches: 120/120 on spec, max T 63.7 C). A wider first window (ratio 5-8, T0 to 58 C,
T_sp to 62 C) put 13 % of batches above the methanol boiling point, which the model does not
represent, and 23 % of fault-free batches off spec through recipe alone.
Plant-model mismatch per batch (hidden from estimators, stored in meta as mm_*): a x [0.75, 1.25]
(log-uniform), k_SE in [0.006, 0.032] L/mol/min (log-uniform), k_H x [0.85, 1.15], c_th x [0.9, 1.1],
UA x [0.9, 1.1].
Outputs (data/): <split>_X.mat/.npz (batches x time x channels), <split>_meta.csv, <split>_ts.csv.gz, run JSON.
Usage: python3 generate_dataset.py [n_train_per_class] [n_test_per_class]
"""
import os, sys, json, time, datetime, hashlib
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import qmc
from scipy.io import savemat
from faults import simulate_batch, CLASSES, DT
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from pilot import AV_2013

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OILS = ["CSFO", "SBFO", "BFVO", "CWVO", "SWVO", "BWVO"]
MEAS = ["T", "T_J", "T_Jin", "T_cmd", "Q_J", "P_st", "pH", "m_ld"]
TRUTH = ["X_E", "cat_act", "OH", "FFA", "W", "A", "fault_active"]
FPAR = ["eta", "phi", "bias", "F_W", "F_A", "a_rel", "dAV"]       # true fault parameters per sample
STATE_NAMES = ["TG", "DG", "MG", "E", "A", "GL", "Cat", "FFA", "W", "Soap", "OH", "V", "T", "T_J", "cat_in"]
SEV = dict(train=[(0.0, 0.2), (0.2, 0.4), (0.6, 0.8)], test_interp=[(0.4, 0.6)], test_extrap=[(0.8, 1.0)])
ONSET = dict(train=[(5, 35), (50, 70)], test_interp=[(35, 50)], test_extrap=[(70, 90)])
IN_BATCH = {"F2", "E1", "E2", "E3", "E4"}


def mismatch(u):
    """Per-batch plant-model mismatch from the fit uncertainty; u in [0,1]^5."""
    return dict(a=float(np.exp(np.log(0.75) + u[0] * np.log(1.25 / 0.75))),   # +/-25 % (fit CI)
                k_SE=float(np.exp(np.log(0.006) + u[1] * np.log(0.032 / 0.006))),  # sensitivity range
                k_H=0.85 + 0.30 * u[2], c_th=0.90 + 0.20 * u[3], UA=0.90 + 0.20 * u[4])


def pick(intervals, u):
    """Map u in [0,1) uniformly onto a union of intervals."""
    L = np.array([b - a for a, b in intervals]); c = np.cumsum(L) / L.sum()
    i = int(np.searchsorted(c, u, side="right")); i = min(i, len(intervals) - 1)
    lo = 0.0 if i == 0 else c[i - 1]
    a, b = intervals[i]
    return a + (u - lo) / (c[i] - lo) * (b - a)


def make_cases(split, n_per_class, seed):
    cases = []
    for ci, cls in enumerate(CLASSES):
        n = n_per_class * (2 if (cls == "N" and split == "train") else 1)
        U = qmc.LatinHypercube(d=16, seed=seed + 101 * ci).random(n)
        for j, u in enumerate(U):
            oil = OILS[int(u[0] * len(OILS))]
            c = dict(split=split, cls=cls, oil=oil, cat_wt=3.5 + 1.5 * u[1], ratio=6 + 2 * u[2],
                     T0_C=50 + 5 * u[3], Tsp_C=57 + 3 * u[4], water_wt=1e-4 + 2e-4 * u[5],
                     AV=AV_2013[oil] * (0.8 + 0.4 * u[6]), s=0.0, t_f=0.0, shape="abrupt", tau_r=20.0,
                     sign=1.0, seed=int(seed * 100000 + ci * 10000 + j), mismatch=mismatch(u[11:16]))
            if cls != "N":
                c["s"] = pick(SEV[split], u[7])
                if cls in IN_BATCH:
                    c["t_f"] = pick(ONSET[split], u[8])
                    c["shape"] = "abrupt" if u[9] < 0.5 else "ramp"
                    c["tau_r"] = 10 + 20 * u[10]
                if cls == "E4":
                    c["sign"] = 1.0 if u[9] * 7 % 1 < 0.5 else -1.0
            cases.append(c)
    return cases


def jco_cases(n, seed):
    U = qmc.LatinHypercube(d=11, seed=seed).random(n)
    return [dict(split="test_jco", cls="F3", oil="JCO", cat_wt=3.5 + 1.5 * u[0], ratio=6 + 2 * u[1],
                 T0_C=50 + 5 * u[2], Tsp_C=57 + 3 * u[3], water_wt=1e-4 + 2e-4 * u[4],
                 AV=AV_2013["JCO"] * (0.8 + 0.4 * u[5]), s=0.0, t_f=0.0, shape="abrupt", tau_r=20.0,
                 sign=1.0, seed=int(seed * 100000 + j), mismatch=mismatch(u[6:11]), _no_inject=True)
            for j, u in enumerate(U)]


def run(c):
    cc = dict(c)
    if cc.pop("_no_inject", False):
        cc["cls"] = "N"                      # Jatropha's own fitted activity is the fault
    try:
        o = simulate_batch(cc, seed=c["seed"])
    except Exception as e:                   # recorded, never silently dropped
        return c, None, str(e)
    if c.get("_no_inject"):
        o["fault_active"][:] = 1.0
    S = np.concatenate([o["state"], np.stack([o[k] for k in FPAR], axis=1)], axis=1)
    return c, (np.stack([o[k] for k in MEAS + TRUTH], axis=1).astype(np.float32), S.astype(np.float32)), None


def main(n_train, n_test):
    t0 = time.time()
    plan = dict(train=(n_train, 1), test_interp=(n_test, 2), test_extrap=(n_test, 3))
    summary = {}
    for split, (n, seed) in list(plan.items()) + [("test_jco", (n_test, 4))]:
        cases = jco_cases(n, seed) if split == "test_jco" else make_cases(split, n, seed)
        with Pool(os.cpu_count()) as pool:
            res = pool.map(run, cases, chunksize=8)
        ok = [(c, a[0]) for c, a, e in res if e is None]
        S = np.stack([a[1] for c, a, e in res if e is None])
        fails = [(c, e) for c, _, e in res if e is not None]
        X = np.stack([a for _, a in ok])
        rows = []
        for c, _ in ok:
            r = {k: v for k, v in c.items() if not k.startswith("_") and k != "mismatch"}
            r.update({f"mm_{k}": v for k, v in c["mismatch"].items()})
            rows.append(r)
        meta = pd.DataFrame(rows)
        meta.insert(0, "batch_id", [f"{split}_{i:05d}" for i in range(len(meta))])
        meta["label"] = meta.cls.map({c: i for i, c in enumerate(CLASSES)})
        X_E = X[:, :, len(MEAS) + TRUTH.index("X_E")]
        meta["XE_peak"] = X_E.max(axis=1); meta["t_peak"] = X_E.argmax(axis=1) * DT
        meta["XE_end"] = X_E[:, -1]; meta["on_spec_peak"] = meta.XE_peak >= 0.965
        meta.to_csv(f"{OUT}/{split}_meta.csv", index=False)
        np.savez_compressed(f"{OUT}/{split}_X.npz", X=X, channels=np.array(MEAS + TRUTH), dt=DT)
        np.savez_compressed(f"{OUT}/{split}_S.npz", S=S, channels=np.array(STATE_NAMES + FPAR), dt=DT)
        savemat(f"{OUT}/{split}_X.mat", dict(X=X, channels=np.array(MEAS + TRUTH, dtype=object), dt=DT,
                                             batch_id=meta.batch_id.values.astype(object)), do_compression=True)
        tt = np.arange(X.shape[1]) * DT
        long = pd.DataFrame(X.reshape(-1, X.shape[2]), columns=MEAS + TRUTH)
        long.insert(0, "t_min", np.tile(tt, X.shape[0]))
        long.insert(0, "batch_id", np.repeat(meta.batch_id.values, X.shape[1]))
        long.to_csv(f"{OUT}/{split}_ts.csv.gz", index=False, float_format="%.5g", compression="gzip")
        summary[split] = dict(n=len(meta), failed=len(fails), fail_examples=[str(e)[:120] for _, e in fails[:3]],
                              per_class=meta.cls.value_counts().to_dict(),
                              on_spec_frac_by_class=meta.groupby("cls").on_spec_peak.mean().round(3).to_dict(),
                              sha1=hashlib.sha1(open(f"{OUT}/{split}_X.npz", "rb").read()).hexdigest())
        print(split, summary[split]["n"], "batches,", len(fails), "failed", flush=True)
    log = dict(step="dataset", run_utc=datetime.datetime.now(datetime.UTC).isoformat(),
               n_train_per_class=n_train, n_test_per_class=n_test, dt=DT, measured=MEAS, truth=TRUTH,
               classes=CLASSES, severity_bands=SEV, onset_windows=ONSET, seconds=round(time.time() - t0, 1),
               splits=summary)
    json.dump(log, open(f"{OUT}/dataset.run.json", "w"), indent=1, default=str)
    print(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250, int(sys.argv[2]) if len(sys.argv) > 2 else 80)
