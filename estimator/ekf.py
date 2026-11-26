"""Baseline: extended Kalman filter on the same plant model.

State (20): 11 species (mol), V, T, T_J, load-cell mass m, and five random-walk fault parameters
[a_rel, eta, bias b, F_W (mol/min), k_loss (1/min)]. Inputs: measured T_J,in and F_J.
Measurements: T + b, T_J, apparent pH, m. Nominal parameters (vegetable-oil group); the hidden
plant-model mismatch is not known to the filter, exactly as for the networks.
Discretisation: state propagated by LSODA over each 0.5 min step; covariance by Phi = expm(A dt) with
A the finite-difference Jacobian of the continuous model. Tuning was set on training batches only.
Usage: python3 ekf.py <split> <n_per_class>   (stratified subset, seeded)
"""
import os, sys, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from scipy.linalg import expm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model")); sys.path.insert(0, os.path.join(HERE, "..", "faults"))
from plant import rhs, IX, I_V, I_T, I_TJ, SP          # noqa: E402
from pilot import pilot_case, PILOT                     # noqa: E402
from faults import PH_REF, C_B_REF, MW_FAME, RHO_FAME   # noqa: E402

DATA = os.path.join(HERE, "..", "data")
NS = len(SP)                      # 11
iV, iT, iTJ, iM = NS, NS + 1, NS + 2, NS + 3
iA, iE, iB, iW, iK = NS + 4, NS + 5, NS + 6, NS + 7, NS + 8
N = NS + 9
DT, STRIDE = 0.5, 2
MEAS = ["T", "T_J", "pH", "m_ld"]
Rm = np.diag([0.05 ** 2, 0.05 ** 2, 0.02 ** 2, 0.2 ** 2])
QSP, QA, QB, QW, QK, QE, T_UPD, QT, QTJ = 1e-4, 0.02, 0.05, 0.5, 5e-4, 0.02, 0.0, 0.3, 1.0   # tuning (set on training batches)
LO = dict(a=0.02, eta=0.05, b=-6.0, W=0.0, K=0.0); HI = dict(a=2.0, eta=1.5, b=6.0, W=200.0, K=0.1)


def f_cont(x, p, a_nom, Tjin, FJ):
    """Continuous dynamics of the augmented state."""
    xp = np.zeros(14 + 1)                        # plant state incl. cat_in
    xp[:NS] = x[:NS]; xp[I_V] = x[iV]; xp[I_T] = x[iT]; xp[I_TJ] = x[iTJ]
    p.a = a_nom * min(max(x[iA], LO["a"]), HI["a"]); p.eta = min(max(x[iE], LO["eta"]), HI["eta"])
    FW = max(x[iW], 0.0); kl = max(x[iK], 0.0)
    d = rhs(0.0, xp, p, lambda t: dict(T_Jin=Tjin, F_J=FJ, F_W=FW, F_A=-kl * max(x[IX["A"]], 0.0)))
    dx = np.zeros(N)
    dx[:NS] = d[:NS]; dx[iV] = d[I_V]; dx[iT] = d[I_T]; dx[iTJ] = d[I_TJ]
    dx[iM] = (-kl * max(x[IX["A"]], 0.0) * 32.042 + FW * 18.015) / 1000.0
    return dx


def h(x):
    V_bp = (max(x[IX["A"]], 1e-9) * 32.042 / 0.791 + max(x[IX["E"]], 0.0) * MW_FAME / RHO_FAME) / 1000.0
    base = max(x[IX["Cat"]] + x[IX["OH"]], 1e-9)
    return np.array([x[iT] + x[iB] - 273.15, x[iTJ] - 273.15, PH_REF + np.log10(base / V_bp / C_B_REF), x[iM]])


def jac(fun, x, eps_rel=1e-6):
    f0 = fun(x); J = np.zeros((len(f0), len(x)))
    for j in range(len(x)):
        e = eps_rel * max(abs(x[j]), 1e-3); xx = x.copy(); xx[j] += e
        J[:, j] = (fun(xx) - f0) / e
    return J


def run_batch(args):
    meta, Y, Tjin = args
    x0p, p = pilot_case(meta["oil"], ratio=meta["ratio"], cat_wt=meta["cat_wt"], AV=meta["AV"],
                        water_wt=meta["water_wt"], T0_C=meta["T0_C"], group="veg")
    a_nom, FJ, nTG0 = p.a, PILOT["F_J_Lmin"], x0p[IX["TG"]]
    x = np.zeros(N); x[:NS] = x0p[:NS]; x[iV] = x0p[I_V]; x[iT] = Y[0, 0] + 273.15; x[iTJ] = Y[0, 1] + 273.15
    x[iM] = Y[0, 3]; x[iA] = 1.0; x[iE] = 1.0
    P = np.diag(np.r_[(1e-3 * nTG0) ** 2 * np.ones(NS), 1e-2, 0.1 ** 2, 0.1 ** 2, 0.3 ** 2,
                      0.3 ** 2, 0.1 ** 2, 1.0, 1.0, 1e-6])
    P[IX["FFA"], IX["FFA"]] = (0.07 * nTG0) ** 2          # acid value unknown within ~4 mg KOH/g
    P[IX["W"], IX["W"]] = (0.05 * nTG0) ** 2
    Q = np.diag(np.r_[(QSP * nTG0) ** 2 * np.ones(NS), 1e-6, QT ** 2, QTJ ** 2, 0.01,
                      QA ** 2, QE ** 2, QB ** 2, QW ** 2, QK ** 2]) * DT
    est = np.zeros((Y.shape[0], 4)); est[0] = [x[IX["E"]] / (3 * nTG0), x[IX["Cat"]] / x[iV], x[iA], x[iE]]
    for k in range(1, Y.shape[0]):
        u = Tjin[k - 1] + 273.15
        fun = lambda xx: f_cont(xx, p, a_nom, u, FJ)
        A = jac(fun, x)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(A)):
            est[k:] = np.nan                    # filter diverged: recorded, not hidden
            return est
        s = solve_ivp(lambda t, xx: fun(xx), (0, DT), x, method="LSODA", rtol=1e-6, atol=1e-9)
        x = s.y[:, -1] if s.success else x + DT * fun(x)
        Phi = expm(A * DT)
        P = Phi @ P @ Phi.T + Q
        if k * DT < T_UPD:
            est[k] = [x[IX["E"]] / (3 * nTG0), x[IX["Cat"]] / x[iV], x[iA], x[iE]]
            continue
        H = jac(h, x)
        S_ = H @ P @ H.T + Rm
        K = P @ H.T @ np.linalg.solve(S_, np.eye(4))
        x = x + K @ (Y[k] - h(x))
        P = (np.eye(N) - K @ H) @ P @ (np.eye(N) - K @ H).T + K @ Rm @ K.T
        x[:NS] = np.maximum(x[:NS], 0.0)
        x[iA] = min(max(x[iA], LO["a"]), HI["a"]); x[iE] = min(max(x[iE], LO["eta"]), HI["eta"])
        x[iB] = min(max(x[iB], LO["b"]), HI["b"]); x[iW] = min(max(x[iW], LO["W"]), HI["W"])
        x[iK] = min(max(x[iK], LO["K"]), HI["K"])
        est[k] = [x[IX["E"]] / (3 * nTG0), x[IX["Cat"]] / x[iV], x[iA], x[iE]]
    return est


def subset(meta, n_per_class, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.where(meta.cls == c)[0], min(n_per_class, (meta.cls == c).sum()), replace=False)
                                   for c in meta.cls.unique()]))


def main(split, n_per_class):
    t0 = time.time()
    X = np.load(f"{DATA}/{split}_X.npz", allow_pickle=True); xc = list(X["channels"])
    meta = pd.read_csv(f"{DATA}/{split}_meta.csv")
    sel = subset(meta, n_per_class)
    Xs = X["X"][sel][:, ::STRIDE, :]
    args = [(meta.iloc[i].to_dict(), Xs[j][:, [xc.index(c) for c in MEAS]].astype(np.float64),
             Xs[j][:, xc.index("T_Jin")].astype(np.float64)) for j, i in enumerate(sel)]
    with Pool(os.cpu_count()) as pool:
        est = pool.map(run_batch, args, chunksize=1)
    est = np.stack(est)
    div = np.isnan(est[:, -1, 0])
    print(split, "diverged:", int(div.sum()), "of", len(sel), "by class:", meta.cls.values[sel][div].tolist()[:20])
    np.savez_compressed(os.path.join(HERE, f"ekf_{split}.npz"), est=est, sel=sel, diverged=div,
                        channels=np.array(["X_E", "cat_act", "a_rel", "eta"]))
    print(split, "EKF batches", len(sel), "in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
