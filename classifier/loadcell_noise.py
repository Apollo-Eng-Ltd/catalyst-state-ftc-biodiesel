"""Detection with a noisier load cell.

Extra Gaussian noise is added to the load-cell residual of every batch so that its total standard deviation
is 0.5 or 1.0 kg (0.2 kg in the base case). Mean, covariance and threshold are re-estimated on the fault-free
training batches exactly as in residuals.py, and the alarm rates on the held-out splits are recomputed.
The model-free load-cell alarm threshold is scaled with the noise (five standard deviations of the
2.5-min moving mean). Output: loadcell_noise.json
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from residuals import first_alarm, DT, RES, DATA

SPLITS = ["train", "test_interp", "test_extrap"]
out = {}
for sd in [0.2, 0.5, 1.0]:
    rng = np.random.default_rng(7)
    R = {}
    for sp in SPLITS:
        r = np.load(f"{DATA}/{sp}_R.npz")["R"].astype(float)
        extra = np.sqrt(max(sd ** 2 - 0.2 ** 2, 0.0))
        r[:, :, 5] += rng.normal(0.0, extra, r[:, :, 5].shape)
        R[sp] = (r, pd.read_csv(f"{DATA}/{sp}_meta.csv"))
    Rt, mt = R["train"]
    i0, w = int(2.0 / DT), int(2.0 / DT)
    Rn = Rt[mt.cls.values == "N"]; nT = Rt.shape[1]
    MU = np.zeros((nT, 6)); SINV = np.zeros((nT, 6, 6))
    for t in range(nT):
        blk = Rn[:, max(0, t - w):t + w + 1, :].reshape(-1, 6)
        MU[t] = blk.mean(0); SINV[t] = np.linalg.inv(np.cov(blk.T) + 1e-8 * np.eye(6))
    def D_of(r):
        Z = r - MU[None]; D = np.einsum("bti,tij,btj->bt", Z, SINV, Z); D[:, :i0] = 0.0; return D
    Dn = D_of(Rn); choice = None
    for k in [1, 2, 4, 8]:
        for q in [0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999]:
            h = float(np.quantile(Dn[:, i0:], q)); fa = (first_alarm(Dn, h, k) >= 0).mean()
            if fa <= 0.05:
                if choice is None or h * k < choice[0] * choice[1]:
                    choice = (h, k)
                break
    h, k = choice
    res = dict(h=h, k=k, mass_alarm_kg=round(5 * sd / np.sqrt(10), 3))
    for sp in ["test_interp", "test_extrap"]:
        r, m = R[sp]
        fa = first_alarm(D_of(r), h, k)
        m = m.assign(det=fa >= 0, t_al=np.where(fa >= 0, fa * DT, np.nan))
        m["delay"] = np.where(m.det & (m.t_al >= m.t_f), m.t_al - m.t_f, np.nan)
        g = m.groupby("cls")
        res[sp] = dict(rate=g.det.mean().round(3).to_dict(), delay=g.delay.median().round(2).to_dict())
    out[str(sd)] = res
    print(sd, h, k, res["test_interp"]["rate"], res["test_extrap"]["rate"], flush=True)
json.dump(out, open(os.path.join(HERE, "loadcell_noise.json"), "w"), indent=1)
