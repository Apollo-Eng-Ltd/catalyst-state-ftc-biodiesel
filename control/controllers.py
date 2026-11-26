"""The three controllers on the stepwise plant.

PID       PI on reactor temperature (T_sp = 60 C) through the TCU; no dosing; batch stopped at a fixed
          time t_stop tuned offline on fault-free batches (recipe practice).
NMPC-nom  economic NMPC (nmpc.py, RK4 internal model). State: nominal model run alongside the plant
          with the known inputs, temperatures reset to the measurements each step. Parameters nominal
          (a_rel = eta = phi = 1, no water ingress or methanol loss).
FT-NMPC   same NMPC; state and fault parameters from the closed-loop causal estimator (est_data_cl);
          the measured temperature is corrected by the estimated sensor bias before use.
Stop rule (both NMPC): stop when X_E(est) >= 0.965 + backoff and the predicted gain over the horizon is < 0.1
point, or when the predicted gain is < 0.05 point (no further progress possible), or at t_max.
Returned per batch: true X_E at stop, on-spec flag, stop time, catalyst and methanol used, peak true
temperature, temperature IAE/ISE/ITAE about 60 C, NMPC solve times.
"""
import os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model")); sys.path.insert(0, os.path.join(HERE, "..", "estimator"))
from plant import IX as PIX, I_V, I_T, I_TJ                     # noqa: E402
from pilot import pilot_case, PILOT                              # noqa: E402
import casadi_plant as cp                                        # noqa: E402
from nmpc import NMPC, DT_C                                      # noqa: E402
from plant_sim import Batch                                      # noqa: E402

SPEC, T_MAX, T_REF_C = 0.965, 120.0, 60.0
EST_MEAS = ["T", "T_J", "T_Jin", "T_cmd", "Q_J", "P_st", "pH", "m_ld", "u_cat", "F_Ain"]
_EST = {}


def estimator(tag="data_cl"):
    if tag not in _EST:
        from train_estimator import Net
        ck = torch.load(os.path.join(HERE, "..", "estimator", f"est_{tag}.pt"), weights_only=False)
        net = Net(ck["n_in"], ck["n_out"]).double(); net.load_state_dict(ck["state"]); net.eval()
        _EST[tag] = (net, ck)
    return _EST[tag]


def estimate(batch, case, tag="data_cl"):
    """Run the causal estimator on everything measured so far; return state (casadi order) and q."""
    net, ck = estimator(tag)
    A = batch.arrays()
    Xm = np.stack([A[k] for k in EST_MEAS], 1)[::2]
    st = np.array([case["cat_wt"], case["ratio"], case["AV"], case["water_wt"], T_REF_C, case["T0_C"]])
    xin = np.concatenate([Xm, np.repeat(st[None], len(Xm), 0)], 1)[None]
    with torch.no_grad():
        y = net(torch.tensor((xin - ck["mu_x"]) / ck["sd_x"])).numpy()[0, -1] * ck["sd_y"] + ck["mu_y"]
    x0p = batch.x0
    nTG0, V0 = x0p[PIX["TG"]], x0p[I_V]
    z, zV, That = y[:11], y[11], y[12]
    eta, phi, b, a_rel, F_W, kl = y[13:19]
    x = np.r_[np.maximum(z, 0) * nTG0, zV * V0, That, A["T_J"][-1] + 273.15]
    q = np.array([np.clip(a_rel, 0.05, 2.0), np.clip(eta, 0.05, 1.5), max(F_W, 0.0), max(kl, 0.0), np.clip(phi, 0.05, 1.0)])
    return x, q, float(b)


def consts(p):
    return dict(rho_cp=p.rho_cp, UA=p.UA, V_J=p.V_J, rho_cp_J=p.rho_cp_J, F_J=PILOT["F_J_Lmin"])


def summarise(batch, t_stop, solve_t=None, extra=None):
    A = batch.arrays()
    k = int(round(t_stop / 0.25))
    t = A["t"][:k + 1]; Tt = A["T_true"][:k + 1]
    e = np.abs(Tt - T_REF_C)
    out = dict(XE_stop=float(A["X_E"][k]), on_spec=bool(A["X_E"][k] >= SPEC), t_stop=float(t_stop),
               cat_charge_kg=float(batch.x0[PIX["Cat"]] * 138.205 / 1000), cat_topup_kg=float(batch.cat_dosed * 138.205 / 1000),
               meoh_makeup_kg=float(batch.meoh_dosed * 32.042 / 1000), T_peak=float(Tt.max()),
               T_viol=bool(Tt.max() > 64.7), IAE=float(np.trapezoid(e, t)), ISE=float(np.trapezoid(e ** 2, t)),
               ITAE=float(np.trapezoid(t * e, t)))
    if solve_t:
        out.update(solve_mean=float(np.mean(solve_t)), solve_max=float(np.max(solve_t)), n_solves=len(solve_t))
    if extra:
        out.update(extra)
    return out


def run_pid(case, t_stop, seed=0):
    b = Batch(case, seed=seed)
    n = int(round(t_stop / DT_C))
    for _ in range(n):
        b.step(dict(mode="pi", T_sp=273.15 + T_REF_C))
    return summarise(b, n * DT_C), b


def run_nmpc(case, mode, seed=0, H=6, est_tag="data_cl", backoff=0.0):
    b = Batch(case, seed=seed)
    x0p, p_nom = pilot_case(case["oil"], ratio=case["ratio"], cat_wt=case["cat_wt"], AV=case["AV"],
                            water_wt=case["water_wt"], T0_C=case["T0_C"], group="veg")
    c = consts(p_nom)
    ctl = NMPC(c, H=H, method="rk", n_fe=40)
    F = cp.make_integrator(c, DT_C)                                   # nominal observer (NMPC-nom)
    TG0, nb0, nA0 = x0p[PIX["TG"]], x0p[PIX["Cat"]], x0p[PIX["A"]]
    xn = np.r_[[x0p[PIX[s]] for s in cp.SP], x0p[I_V], x0p[I_T], x0p[I_TJ]]
    u_prev = np.array([273.15 + T_REF_C, 0.0, 0.0])
    solve_t, decisions = [], []
    t, stopped = 0.0, False
    q_nom = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    while t < T_MAX - 1e-9:
        A = b.arrays()
        if mode == "nominal":
            xn[cp.iT], xn[cp.iTJ] = A["T"][-1] + 273.15, A["T_J"][-1] + 273.15
            x, q = xn.copy(), q_nom
        else:
            x, q, bias = estimate(b, case, est_tag)
        XE_now = x[cp.IX["E"]] / (3 * TG0)
        n_act = int(min(H, round((T_MAX - t) / DT_C)))
        u, info = ctl.solve(x, q, TG0, nb0, nA0, u_prev, n_act)
        solve_t.append(info["t_wall"])
        gain = info["XE_pred_end"] - XE_now
        decisions.append(dict(t=t, XE_est=XE_now, gain=gain, T_cmd=u[0] - 273.15, u_cat=u[1], F_A=u[2],
                              q=np.round(q, 3).tolist(), ok=info["success"]))
        if t > 0 and ((XE_now >= SPEC + backoff and gain < 1e-3) or gain < 5e-4):
            stopped = True
            break
        b.step(dict(mode="direct", T_cmd=float(u[0]), u_cat=float(u[1]), F_A=float(u[2])))
        if mode == "nominal":
            xn = np.array(F(x0=xn, p=np.r_[u, q_nom])["xf"]).ravel()
        u_prev = u; t += DT_C
    return summarise(b, t, solve_t, dict(stopped_by_rule=stopped, decisions=decisions)), b
