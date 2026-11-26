"""Stop-rule margins for the EKF baseline, same definition and data as calibrate_stoprule.py,
on a stratified subset of cl_test_interp (10 batches per class, seed 0), EKF run online with dosing."""
import os, sys, json
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from controllers2 import EKFOnline
DATA = os.path.join(HERE, "..", "data")
_X = np.load(f"{DATA}/cl_test_interp_X.npz", allow_pickle=True); _S = np.load(f"{DATA}/cl_test_interp_S.npz", allow_pickle=True)
xc, sc = list(_X["channels"]), list(_S["channels"])
X = {"X": _X["X"]}; S = {"S": _S["S"]}
meta = pd.read_csv(f"{DATA}/cl_test_interp_meta.csv")
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(meta.cls == c)[0], 10, replace=False) for c in sorted(meta.cls.unique())])


def one(i):
    A = {c: X["X"][i, :, xc.index(c)].astype(float) for c in xc}
    A["t"] = np.arange(X["X"].shape[1]) * 0.25
    case = meta.iloc[i].to_dict()
    ekf = EKFOnline(case, {k: v[:1] for k, v in A.items()})
    nTG0 = S["S"][i, 0, sc.index("TG")]
    xh = np.full(241, np.nan)
    xh[0] = 0.0
    for j in range(1, 241):
        Aj = {k: v[:2 * j + 1] for k, v in A.items()}
        if not ekf.update(Aj):
            break
        xh[j] = ekf.x[3] / (3 * nTG0)
    xe = S["S"][i, ::2, sc.index("E")] / (3 * nTG0)
    return i, xh, xe


if __name__ == "__main__":
    with Pool(2) as p:
        R = p.map(one, idx)
    t = np.arange(241) * 0.5
    E_, near0, near1 = [], [], []
    for i, xh, xe in R:
        e = xh - xe; c = meta.cls[i]; tf = meta.t_f[i]
        nr = (xh > 0.955) & (xh < 0.99) & (t >= 10) & np.isfinite(xh)
        seen = (c in ("F2", "E3", "E1", "E2")) & (t >= tf)
        near0.append(e[nr & ~seen]); near1.append(e[nr & seen])
    e0, e1 = np.concatenate(near0), np.concatenate(near1)
    q = lambda v: float(np.sort(v)[min(int(np.ceil((len(v) + 1) * 0.95)) - 1, len(v) - 1)])
    out = dict(delta0=q(e0), delta1=q(e1), n0=len(e0), n1=len(e1), batches=len(idx), split="cl_test_interp subset",
               estimator="EKF", diverged=int(sum(np.isnan(r[1]).any() for r in R)),
               bias_pp=float(100 * np.concatenate([e0, e1]).mean()), sd_pp=float(100 * np.concatenate([e0, e1]).std()))
    json.dump(out, open(os.path.join(HERE, "stoprule_calibration_ekf.json"), "w"), indent=1)
    print(out)
