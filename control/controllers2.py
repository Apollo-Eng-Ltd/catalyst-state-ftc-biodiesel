"""Calibrated, alarm-aware stop rule (v2) and an EKF-based FT-NMPC baseline.

Stop rule v1: stop when X_E,hat >= 0.965 + backoff and predicted gain < 0.1 point, or when the
predicted gain < 0.05 point. Diagnosis of the first Monte Carlo (seed 21): all five off-spec FT-NMPC batches were
stopped by the second clause with a negative predicted gain while the true conversion was still rising.
Stop rule v2:
  C1 (on specification)  X_E,hat - delta >= 0.965 and predicted gain < 0.1 point
  C2 (peak reached)      predicted gain < 0.05 point and the estimate itself stopped rising over the last
                         interval (X_E,hat(t) - X_E,hat(t-5) < 0.05 point)
  C3 (peak passed)       the estimate fell at two consecutive decisions
  stop when C1 or C2 holds at two consecutive decisions (5-min confirmation), when C3 holds, or at t_max.
  delta = delta0 (0.67 point) without alarm, delta1 (1.03 points) once a model-free alarm has fired;
  both are 95 % one-sided quantiles of the estimator error on held-out closed-loop batches
  (calibrate_stoprule.py, cl_test_interp); not tuned on any Monte Carlo set.
Model-free alarms (latched; 2.5-min moving means, thresholds about 5 noise standard deviations):
  load cell   |m_ld - (m0 + dosed mass)| > 0.32 kg       water ingress, methanol loss
  TCU         |T_cmd - T_Jin| > 0.25 K                   temperature-unit fault
  stirrer     |P_st / P0 - 1| > 0.03                     agitation loss
EKF-NMPC: same NMPC and rule v2; state and fault parameters from an online EKF (estimator/ekf.py model,
tuning and noise, extended with the dosing inputs); TCU capacity phi by least squares on the last 10 min
of (T_Jin - T_J) against (T_cmd - T_J). Its margins come from the same calibration on the EKF
(calibrate_stoprule_ekf.py) when available, otherwise the network's.
"""
import os, sys, json, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model")); sys.path.insert(0, os.path.join(HERE, "..", "estimator"))
sys.path.insert(0, os.path.join(HERE, "..", "faults"))
from plant import IX as PIX, I_V, I_T, I_TJ, rhs              # noqa: E402
from pilot import pilot_case, PILOT                            # noqa: E402
import casadi_plant as cp                                      # noqa: E402
from nmpc import NMPC, DT_C                                    # noqa: E402
from plant_sim import Batch                                    # noqa: E402
from controllers import estimate, consts, summarise, SPEC, T_MAX  # noqa: E402
from faults import P0_W                                        # noqa: E402
import ekf as E                                                # noqa: E402

CAL = json.load(open(os.path.join(HERE, "stoprule_calibration.json")))
_cal_ekf = os.path.join(HERE, "stoprule_calibration_ekf.json")
CAL_EKF = json.load(open(_cal_ekf)) if os.path.exists(_cal_ekf) else CAL
W_DEF = dict(X=100.0, cat=5.0, A=1.0, du=0.05, s=1e3)


def alarms(A, m0):
    """Model-free alarms on the measurements logged so far; returns dict of booleans (any time so far)."""
    n = len(A["t"])
    if n < 10:
        return dict(mass=False, tcu=False, stir=False)
    dosed = np.concatenate([[0.0], np.cumsum((A["u_cat"][1:] * 138.205 + A["F_Ain"][1:] * 32.042) / 1000.0 * 0.25)])
    rm = A["m_ld"] - (m0 + dosed)
    rt = A["T_cmd"] - A["T_Jin"]
    rp = A["P_st"] / P0_W - 1.0
    mv = lambda r: np.convolve(r, np.ones(10) / 10, mode="valid")
    return dict(mass=bool(np.any(np.abs(mv(rm)) > 0.32)), tcu=bool(np.any(np.abs(mv(rt)) > 0.25)),
                stir=bool(np.any(np.abs(mv(rp)) > 0.03)))


class EKFOnline:
    """Online version of estimator/ekf.py (same model, tuning, noise), with the dosing inputs."""
    def __init__(self, case, A0):
        x0p, p = pilot_case(case["oil"], ratio=case["ratio"], cat_wt=case["cat_wt"], AV=case["AV"],
                            water_wt=case["water_wt"], T0_C=case["T0_C"], group="veg")
        self.p, self.a_nom, self.FJ, self.nTG0 = p, p.a, PILOT["F_J_Lmin"], x0p[PIX["TG"]]
        N, NS = E.N, E.NS
        x = np.zeros(N); x[:NS] = x0p[:NS]; x[E.iV] = x0p[I_V]
        x[E.iT] = A0["T"][0] + 273.15; x[E.iTJ] = A0["T_J"][0] + 273.15; x[E.iM] = A0["m_ld"][0]
        x[E.iA] = 1.0; x[E.iE] = 1.0
        P = np.diag(np.r_[(1e-3 * self.nTG0) ** 2 * np.ones(NS), 1e-2, 0.1 ** 2, 0.1 ** 2, 0.3 ** 2,
                          0.3 ** 2, 0.1 ** 2, 1.0, 1.0, 1e-6])
        P[PIX["FFA"], PIX["FFA"]] = (0.07 * self.nTG0) ** 2
        P[PIX["W"], PIX["W"]] = (0.05 * self.nTG0) ** 2
        self.Q = np.diag(np.r_[(E.QSP * self.nTG0) ** 2 * np.ones(NS), 1e-6, E.QT ** 2, E.QTJ ** 2, 0.01,
                               E.QA ** 2, E.QE ** 2, E.QB ** 2, E.QW ** 2, E.QK ** 2]) * E.DT
        self.x, self.P, self.k = x, P, 0                     # k: index (0.25-min grid) of last update
        self.ok = True

    def f(self, xx, Tjin, ucat, FA):
        x = xx; p = self.p; NS = E.NS
        xp = np.zeros(15); xp[:NS] = x[:NS]; xp[I_V] = x[E.iV]; xp[I_T] = x[E.iT]; xp[I_TJ] = x[E.iTJ]
        p.a = self.a_nom * min(max(x[E.iA], E.LO["a"]), E.HI["a"]); p.eta = min(max(x[E.iE], E.LO["eta"]), E.HI["eta"])
        FW = max(x[E.iW], 0.0); kl = max(x[E.iK], 0.0)
        d = rhs(0.0, xp, p, lambda t: dict(T_Jin=Tjin, F_J=self.FJ, F_W=FW, F_A=FA - kl * max(x[PIX["A"]], 0.0), u_cat=ucat))
        dx = np.zeros(E.N)
        dx[:NS] = d[:NS]; dx[E.iV] = d[I_V]; dx[E.iT] = d[I_T]; dx[E.iTJ] = d[I_TJ]
        dx[E.iM] = ((FA - kl * max(x[PIX["A"]], 0.0)) * 32.042 + FW * 18.015 + ucat * 138.205) / 1000.0
        return dx

    def update(self, A):
        """Process all new 0.5-min samples in the batch log."""
        kmax = len(A["t"]) - 1
        while self.ok and self.k + 2 <= kmax:
            k0, k1 = self.k, self.k + 2
            Tjin = A["T_Jin"][k0] + 273.15; ucat = A["u_cat"][k1]; FA = A["F_Ain"][k1]
            fun = lambda xx: self.f(xx, Tjin, ucat, FA)
            Aj = E.jac(fun, self.x)
            if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(Aj)):
                self.ok = False; break
            s = solve_ivp(lambda t, xx: fun(xx), (0, E.DT), self.x, method="LSODA", rtol=1e-6, atol=1e-9)
            x = s.y[:, -1] if s.success else self.x + E.DT * fun(self.x)
            Phi = expm(Aj * E.DT)
            P = Phi @ self.P @ Phi.T + self.Q
            Y = np.array([A["T"][k1], A["T_J"][k1], A["pH"][k1], A["m_ld"][k1]])
            H = E.jac(E.h, x)
            S_ = H @ P @ H.T + E.Rm
            K = P @ H.T @ np.linalg.solve(S_, np.eye(4))
            x = x + K @ (Y - E.h(x))
            IKH = np.eye(E.N) - K @ H
            P = IKH @ P @ IKH.T + K @ E.Rm @ K.T
            x[:E.NS] = np.maximum(x[:E.NS], 0.0)
            for i, kk in [(E.iA, "a"), (E.iE, "eta"), (E.iB, "b"), (E.iW, "W"), (E.iK, "K")]:
                x[i] = min(max(x[i], E.LO[kk]), E.HI[kk])
            self.x, self.P, self.k = x, P, k1
        return self.ok

    def state(self, A):
        x = self.x
        b = A["T_cmd"][-40:] - A["T_J"][-40:]; a = A["T_Jin"][-40:] - A["T_J"][-40:]
        phi = float(np.clip(np.dot(a, b) / np.dot(b, b), 0.05, 1.0)) if np.dot(b, b) > 40 * 1.0 else 1.0
        xs = np.r_[[x[PIX[s]] for s in cp.SP], x[E.iV], x[E.iT], A["T_J"][-1] + 273.15]
        q = np.array([min(max(x[E.iA], 0.05), 2.0), min(max(x[E.iE], 0.05), 1.5), max(x[E.iW], 0.0), max(x[E.iK], 0.0), phi])
        return xs, q


def oracle_state(b, p_nom):
    """True plant state and fault parameters in the NMPC's coordinates (perfect-information benchmark)."""
    x = b.z[:-1]
    phi, nu, _, F_W, F_loss = b.faults_at(b.t, x)
    kl = F_loss / max(x[PIX["A"]], 1e-9)
    xs = np.r_[[x[PIX[s]] for s in cp.SP], x[I_V], x[I_T], x[I_TJ]]
    q = np.array([b.p.a / p_nom.a, nu, F_W, kl, phi])
    return xs, q


def run_nmpc2(case, est="gru", rule="v2", seed=0, H=6, w=None, backoff=0.0, thr=None, cal_override=None, log_truth=False):
    thr = dict(dict(g1=1e-3, g2=5e-4, rise=5e-4, confirm=True), **(thr or {}))
    b = Batch(case, seed=seed)
    x0p, p_nom = pilot_case(case["oil"], ratio=case["ratio"], cat_wt=case["cat_wt"], AV=case["AV"],
                            water_wt=case["water_wt"], T0_C=case["T0_C"], group="veg")
    c = consts(p_nom)
    ctl = NMPC(c, H=H, method="rk", n_fe=40, w=w or W_DEF)
    TG0, nb0, nA0 = x0p[PIX["TG"]], x0p[PIX["Cat"]], x0p[PIX["A"]]
    cal = cal_override or (CAL_EKF if est == "ekf" else CAL)
    ekf = EKFOnline(case, b.arrays()) if est == "ekf" else None
    u_prev = np.array([273.15 + 60.0, 0.0, 0.0])
    solve_t, decisions = [], []
    t, stopped, cond_prev, XE_prev, XE_prev2, alarm, ekf_fail = 0.0, False, False, None, None, False, False
    while t < T_MAX - 1e-9:
        A = b.arrays()
        if est == "gru":
            x, q, _ = estimate(b, case, "data_cl")
        elif est == "oracle":
            x, q = oracle_state(b, p_nom)
        else:
            if not ekf.update(A):
                ekf_fail = True
            x, q = ekf.state(A)
        XE_now = x[cp.IX["E"]] / (3 * TG0)
        n_act = int(min(H, round((T_MAX - t) / DT_C)))
        u, info = ctl.solve(x, q, TG0, nb0, nA0, u_prev, n_act)
        solve_t.append(info["t_wall"])
        gain = info["XE_pred_end"] - XE_now
        al = alarms(A, b.m0); alarm = alarm or any(al.values())
        decisions.append(dict(t=t, XE_est=XE_now, gain=gain, T_cmd=u[0] - 273.15, u_cat=u[1], F_A=u[2],
                              q=np.round(q, 3).tolist(), ok=info["success"], alarm=alarm,
                              XE_true=float(A["X_E"][-1])))
        if rule == "v1":
            stop = t > 0 and ((XE_now >= SPEC + backoff and gain < 1e-3) or gain < 5e-4)
        else:
            delta = max(cal["delta1"], cal["delta0"]) if alarm else cal["delta0"]
            c1 = XE_now - delta >= SPEC and gain < thr["g1"]
            c2 = gain < thr["g2"] and XE_prev is not None and XE_now - XE_prev < thr["rise"]
            c3 = XE_prev is not None and XE_prev2 is not None and XE_now < XE_prev < XE_prev2
            cond = t > 0 and (c1 or c2)
            stop = (cond and (cond_prev or not thr["confirm"])) or c3
            cond_prev = cond
        if stop:
            stopped = True
            break
        XE_prev2, XE_prev = XE_prev, XE_now
        b.step(dict(mode="direct", T_cmd=float(u[0]), u_cat=float(u[1]), F_A=float(u[2])))
        u_prev = u; t += DT_C
    A = b.arrays()
    k = int(round(t / 0.25))
    QJ = A["Q_J"][:k + 1]
    extra = dict(stopped_by_rule=stopped, decisions=decisions, alarm=alarm, ekf_fail=ekf_fail,
                 heat_kWh=float(np.trapezoid(np.maximum(QJ, 0), A["t"][:k + 1]) / 60.0),
                 cool_kWh=float(np.trapezoid(np.maximum(-QJ, 0), A["t"][:k + 1]) / 60.0))
    return summarise(b, t, solve_t, extra), b
