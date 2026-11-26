"""Model-based residual generator and chi-square detection.

For every batch, the NOMINAL model (fitted parameters; recipe and certificate AV/water from the
batch record; no knowledge of faults or of the hidden plant-model mismatch) is driven open-loop by
the measured jacket inlet temperature. Residuals:
  r_T    = T_meas  - T_hat          reactor temperature
  r_TJ   = T_J,meas - T_J,hat       jacket outlet temperature
  r_P    = (P_st - P0) / P0         stirrer power
  r_TCU  = T_cmd - T_Jin,meas       temperature-unit tracking error
  r_pH   = pH_meas - pH_hat         apparent pH (free base in methanol/ester phase)
  r_m    = m_meas - m_hat           load-cell mass (nominal: constant charge mass)
Detection: D(t) = (r - mu(t))' Sigma(t)^-1 (r - mu(t)); mu(t), Sigma(t) from fault-free TRAINING batches,
pooled over a +/- 2 min window around each sample (residual spread is large during the exotherm, where
plant-model mismatch acts, and small afterwards).
Alarm when D exceeds h for k consecutive samples. h and k are set on fault-free TRAINING batches
only (batch-level false-alarm target 5 %), then frozen for the test splits.
Outputs: data/<split>_R.npz (residuals, D), classifier/detection.run.json
"""
import os, sys, json, datetime
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model")); sys.path.insert(0, os.path.join(HERE, "..", "faults"))
from plant import rhs, I_T, I_TJ                       # noqa: E402
from pilot import pilot_case, PILOT                     # noqa: E402
from faults import P0_W, DT, PH_REF, C_B_REF, MW_FAME, RHO_FAME   # noqa: E402
from plant import IX                                     # noqa: E402

DATA = os.path.join(HERE, "..", "data")
SPLITS = ["train", "test_interp", "test_extrap", "test_jco"]
RES = ["r_T", "r_TJ", "r_P", "r_TCU", "r_pH", "r_m"]


def nominal_prediction(args):
    meta, Tjin_meas = args
    x0, p = pilot_case(meta["oil"], ratio=meta["ratio"], cat_wt=meta["cat_wt"], AV=meta["AV"],
                       water_wt=meta["water_wt"], T0_C=meta["T0_C"], group="veg")  # nominal = standard oil
    t = np.arange(len(Tjin_meas)) * DT
    Tin = Tjin_meas + 273.15
    u = lambda tt: dict(T_Jin=float(np.interp(tt, t, Tin)), F_J=PILOT["F_J_Lmin"])
    s = solve_ivp(rhs, (0, t[-1]), x0, args=(p, u), method="LSODA", t_eval=t, rtol=1e-6, atol=1e-9,
                  max_step=DT)
    if not s.success:
        return None
    Y = s.y
    V_bp = (Y[IX["A"]] * 32.042 / 0.791 + Y[IX["E"]] * MW_FAME / RHO_FAME) / 1000.0
    pH = PH_REF + np.log10(np.maximum(Y[IX["Cat"]] + Y[IX["OH"]], 1e-12) / V_bp / C_B_REF)
    m0 = 600.0 + x0[IX["A"]] * 32.042 / 1000.0 + x0[IX["Cat"]] * 138.205 / 1000.0
    return np.stack([Y[I_T] - 273.15, Y[I_TJ] - 273.15, pH, np.full(Y.shape[1], m0)], axis=1)


def residuals(split):
    d = np.load(f"{DATA}/{split}_X.npz", allow_pickle=True)
    X, ch = d["X"], list(d["channels"])
    meta = pd.read_csv(f"{DATA}/{split}_meta.csv")
    args = [(meta.iloc[i].to_dict(), X[i, :, ch.index("T_Jin")]) for i in range(len(meta))]
    with Pool(os.cpu_count()) as pool:
        pred = pool.map(nominal_prediction, args, chunksize=8)
    R = np.zeros((X.shape[0], X.shape[1], len(RES)), dtype=np.float32)
    for i, yh in enumerate(pred):
        if yh is None:
            raise RuntimeError(f"nominal model failed for {meta.batch_id[i]}")
        R[i, :, 0] = X[i, :, ch.index("T")] - yh[:, 0]
        R[i, :, 1] = X[i, :, ch.index("T_J")] - yh[:, 1]
        R[i, :, 2] = (X[i, :, ch.index("P_st")] - P0_W) / P0_W
        R[i, :, 3] = X[i, :, ch.index("T_cmd")] - X[i, :, ch.index("T_Jin")]
        R[i, :, 4] = X[i, :, ch.index("pH")] - yh[:, 2]
        R[i, :, 5] = X[i, :, ch.index("m_ld")] - yh[:, 3]
    return R, meta, X, ch


def stat(R, Sinv):
    return np.einsum("bti,ij,btj->bt", R, Sinv, R)


def first_alarm(D, h, k):
    """Index of the first sample of the first run of k samples above h, or -1."""
    above = D > h
    out = np.full(D.shape[0], -1)
    for b in range(D.shape[0]):
        run = 0
        for t in range(D.shape[1]):
            run = run + 1 if above[b, t] else 0
            if run == k:
                out[b] = t - k + 1
                break
    return out


def fa_cost(h, k):
    """Rank (h, k) pairs: lower threshold x persistence = earlier, more sensitive alarms."""
    return h * k


def main():
    allR = {}
    for sp in SPLITS:
        R, meta, X, ch = residuals(sp)
        allR[sp] = (R, meta)
        print(sp, "residuals", R.shape, flush=True)
    R, meta = allR["train"]
    i0 = int(2.0 / DT)
    Rn = R[meta.cls.values == "N"]
    nT, w = R.shape[1], int(2.0 / DT)
    MU = np.zeros((nT, len(RES))); SINV = np.zeros((nT, len(RES), len(RES)))
    for t in range(nT):
        blk = Rn[:, max(0, t - w):t + w + 1, :].reshape(-1, len(RES))
        MU[t] = blk.mean(0)
        SINV[t] = np.linalg.inv(np.cov(blk.T) + 1e-8 * np.eye(len(RES)))
    def D_of(R):
        Z = R - MU[None]
        D = np.einsum("bti,tij,btj->bt", Z, SINV, Z)
        D[:, :i0] = 0.0
        return D
    Dn = D_of(Rn)
    choice = None
    for k in [1, 2, 4, 8]:
        for q in [0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999]:
            h = float(np.quantile(Dn[:, i0:], q))
            fa = (first_alarm(Dn, h, k) >= 0).mean()
            if fa <= 0.05:
                if choice is None or fa_cost(h, k) < fa_cost(*choice[:2]):
                    choice = (h, k, float(fa))
                break
    h, k, fa_train = choice
    mu, Sigma = MU.mean(0), np.linalg.inv(SINV.mean(0))
    report = dict(step="detection", run_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                  residuals=RES, mu=mu.tolist(), Sigma=Sigma.tolist(), h=h, k=k,
                  h_chi2_quantile=float(chi2.cdf(h, len(RES))), train_normal_batch_FAR=fa_train, splits={})
    for sp, (R, meta) in allR.items():
        D = D_of(R)
        fa_idx = first_alarm(D, h, k)
        t_al = np.where(fa_idx >= 0, fa_idx * DT, np.nan)
        meta = meta.copy()
        meta["t_alarm"] = t_al
        meta["detected"] = fa_idx >= 0
        meta["alarm_before_onset"] = (meta.t_alarm < meta.t_f) & meta.detected & (meta.cls != "N")
        meta["delay"] = np.where(meta.detected & ~meta.alarm_before_onset, meta.t_alarm - meta.t_f, np.nan)
        np.savez_compressed(f"{DATA}/{sp}_R.npz", R=R, D=D, MU=MU, SINV=SINV, channels=np.array(RES), dt=DT)
        meta[["batch_id", "cls", "t_f", "s", "t_alarm", "detected", "alarm_before_onset", "delay"]].to_csv(
            f"{DATA}/{sp}_detection.csv", index=False)
        g = meta.groupby("cls")
        report["splits"][sp] = dict(
            detect_rate=g.detected.mean().round(3).to_dict(),
            alarm_before_onset=g.alarm_before_onset.mean().round(3).to_dict(),
            median_delay_min=g.delay.median().round(2).to_dict(),
            normal_FAR=float(meta[meta.cls == "N"].detected.mean()) if (meta.cls == "N").any() else None)
        print(sp, json.dumps(report["splits"][sp]), flush=True)
    json.dump(report, open(os.path.join(HERE, "detection.run.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
