"""Fault library and measurement model.

A batch = pilot charge + PI jacket-inlet controller (setpoint T_sp) + at most one fault.
Each fault is a parameter change with onset t_f, severity s in [0, 1] (mapped to a physical
magnitude below) and shape: abrupt (step) or incipient (linear ramp over tau_r minutes).
Feedstock faults F1 and F3 are present in the charge (t_f = 0, no shape).

Classes
  0 N   normal
  1 F1  acid value above certificate     dAV   = 0.5 + 3.5 s  mg KOH/g, in the charge
  2 F2  water ingress during the batch   rate  = (0.02 + 0.28 s) wt% of oil per 10 min (mol/min)
  3 F3  degraded feedstock activity      a    *= 1 - 0.85 s  (Jatropha fitted at 0.12 x vegetable oils)
  4 E1  jacket TCU degradation           T_Jin = T_J + phi (T_Jin,cmd - T_J), phi = 1 - 0.9 s
  5 E2  agitation loss (impeller speed)  nu = N/N0 = 1 - 0.8 s;  eta = nu;  P_stir = P0 nu^3
  6 E3  methanol loss (condenser fault)  F_A = -(0.002 + 0.028 s) n_A per min
  7 E4  reactor temperature sensor bias  b = +/-(0.5 + 3.5 s) K on the measured and controlled T

Measurement model (sample time 0.25 min, Gaussian noise, seeded):
  T     reactor temperature (+ bias b)            sigma 0.05 K
  T_J   jacket outlet temperature                 sigma 0.05 K
  T_Jin jacket inlet temperature (TCU outlet)     sigma 0.05 K
  T_cmd TCU setpoint from the controller          exact (a controller signal)
  Q_J   jacket duty F_J rho cp (T_Jin - T_J), kW  computed from the noisy temperatures
  P_st  stirrer power, W                          sigma 2 % of P0
  pH    apparent pH of the methanol/ester phase   sigma 0.02
        pH = PH_REF + log10(c_b / C_B_REF), c_b = (methoxide + hydroxide) / (V_MeOH + V_ester).
        Free base is diluted into the ester as conversion proceeds (Clark et al. 2013, Bioresour
        Technol 136:771, attribute the pH fall to dilution of OH by products) and lowered by any
        base consumed. Anchor: peak pH 12.0 for 1 wt% K2CO3 at 6:1 (Nyepetsi et al. 2021,
        S Afr J Chem 74:42), i.e. C_B_REF = base of 1 g K2CO3 in the methanol for 100 g oil at 6:1.
  m_ld  load-cell mass of reactor contents, kg    sigma 0.2 kg (about 0.02 % of 1 t full scale)
        changes only through methanol loss and water ingress.
Not measured online (ground-truth labels): X_E, active catalyst, hydroxide, FFA, water, methanol.
"""
import os, sys
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
from plant import rhs, X_E, IX, I_T, I_TJ, I_V, algebraic   # noqa: E402
from pilot import pilot_case, PILOT, AV_2013                    # noqa: E402

PH_REF = 12.0
C_B_REF = (1.0 / 138.205) / (6 * 100.0 / 870.0 * 32.042 / 0.791 / 1000.0)   # mol/L
MW_FAME, RHO_FAME = 292.0, 0.875      # g/mol, kg/L (soybean FAME)
CLASSES = ["N", "F1", "F2", "F3", "E1", "E2", "E3", "E4"]
DT = 0.25
P0_W = 5.0 * 880.0 * 2.0 ** 3 * 0.33 ** 5      # Rushton Np = 5, D = 0.33 m, N = 2 s^-1 (stated design)
KC, TAUI = 6.0, 8.0                              # PI on T_Jin,cmd (tuned in test_pilot.py)


def magnitude(cls, s, sign=1.0):
    return {
        "F1": 0.5 + 3.5 * s,
        "F2": 0.02 + 0.28 * s,
        "F3": 1.0 - 0.85 * s,
        "E1": 1.0 - 0.9 * s,
        "E2": 1.0 - 0.8 * s,
        "E3": 0.002 + 0.028 * s,
        "E4": sign * (0.5 + 3.5 * s),
    }.get(cls, 0.0)


def ramp(t, t_f, shape, tau_r):
    """Fault activation in [0, 1]."""
    if t < t_f:
        return 0.0
    return 1.0 if shape == "abrupt" else min(1.0, (t - t_f) / tau_r)


def simulate_batch(case, t_end=120.0, seed=0, return_states=False):
    """case: dict(oil, ratio, cat_wt, AV_offset, water_wt, T0_C, Tsp_C, cls, s, sign, t_f, shape, tau_r)."""
    cls = case.get("cls", "N")
    m = magnitude(cls, case.get("s", 0.0), case.get("sign", 1.0))
    AV0 = case.get("AV", AV_2013[case["oil"]])
    if cls == "F1":
        AV0 = AV0 + m
    x0, p = pilot_case(case["oil"], ratio=case.get("ratio", 6.0), cat_wt=case.get("cat_wt", 4.0),
                       AV=AV0, water_wt=case.get("water_wt", 0.0002), T0_C=case.get("T0_C", 55.0))
    # plant-model mismatch (hidden from estimators): multipliers on fitted parameters
    mm = case.get("mismatch", {})
    p.a *= mm.get("a", 1.0); p.k_H *= mm.get("k_H", 1.0); p.c_th *= mm.get("c_th", 1.0)
    p.UA *= mm.get("UA", 1.0)
    if "k_SE" in mm:
        p.k_SE = mm["k_SE"]
    if cls == "F3":
        p.a *= m
    a_nom, Tsp = p.a, case.get("Tsp_C", 60.0) + 273.15
    t_f, shape, tau_r = case.get("t_f", 0.0), case.get("shape", "abrupt"), case.get("tau_r", 20.0)
    lo, hi, FJ = PILOT["T_Jin_min_K"], PILOT["T_Jin_max_K"], PILOT["F_J_Lmin"]
    m_oil_g = PILOT["m_oil_kg"] * 1000.0

    def faults_at(t, x):
        r = ramp(t, t_f, shape, tau_r) if cls in ("F2", "E1", "E2", "E3", "E4") else 0.0
        phi = 1.0 - r * (1.0 - m) if cls == "E1" else 1.0
        nu = 1.0 - r * (1.0 - m) if cls == "E2" else 1.0
        b = r * m if cls == "E4" else 0.0
        F_W = r * (m / 100.0 * m_oil_g / 18.015) / 10.0 if cls == "F2" else 0.0
        F_A = -r * m * x[IX["A"]] if cls == "E3" else 0.0
        return phi, nu, b, F_W, F_A

    def tcmd(x, e_int, b):
        return Tsp + KC * ((Tsp - (x[I_T] + b)) + e_int / TAUI)

    def f(t, z):
        x, e_int = z[:-1], z[-1]
        phi, nu, b, F_W, F_A = faults_at(t, x)
        p.eta = nu
        raw = tcmd(x, e_int, b)
        Tc = min(max(raw, lo), hi)
        Tin = x[I_TJ] + phi * (Tc - x[I_TJ])
        dx = rhs(t, x, p, lambda tt: dict(T_Jin=Tin, F_J=FJ, F_W=F_W, F_A=F_A))
        # back-calculation anti-windup (tracking time = tau_I): continuous, so no solver chattering
        return np.append(dx, (Tsp - (x[I_T] + b)) + (Tc - raw) / KC)

    t = np.arange(0.0, t_end + 1e-9, DT)
    sol = solve_ivp(f, (0.0, t_end), np.append(x0, 0.0), method="LSODA", t_eval=t,
                    rtol=1e-6, atol=1e-9, max_step=1.0)
    if not sol.success:
        raise RuntimeError(sol.message)
    rng = np.random.default_rng(seed)
    Z = sol.y
    out = {k: np.empty(len(t)) for k in ["T", "T_J", "T_Jin", "T_cmd", "Q_J", "P_st",
                                         "X_E", "cat_act", "OH", "FFA", "W", "A", "fault_active",
                                         "eta", "phi", "bias", "F_W", "F_A", "pH", "m_ld"]}
    out["state"] = Z[:-1, :].T.copy()            # full plant state (moles, V, T, T_J, cat dosed)
    for k, tk in enumerate(t):
        x, e_int = Z[:-1, k], Z[-1, k]
        phi, nu, b, F_W, F_A = faults_at(tk, x)
        out["eta"][k], out["phi"][k], out["bias"][k], out["F_W"][k], out["F_A"][k] = nu, phi, b, F_W, F_A
        raw = tcmd(x, e_int, b)
        Tc = min(max(raw, lo), hi)
        Tin = x[I_TJ] + phi * (Tc - x[I_TJ])
        c, _, _, cat = algebraic(x, p)
        out["T"][k] = x[I_T] + b
        out["T_J"][k] = x[I_TJ]
        out["T_Jin"][k] = Tin
        out["T_cmd"][k] = Tc
        out["P_st"][k] = P0_W * nu ** 3
        out["X_E"][k] = X_E(x, x0)
        out["cat_act"][k] = cat
        out["OH"][k] = c[IX["OH"]]
        out["FFA"][k] = c[IX["FFA"]]
        out["W"][k] = c[IX["W"]]
        out["A"][k] = c[IX["A"]]
        out["fault_active"][k] = float(cls != "N" and (tk >= t_f))
        V_bp = (x[IX["A"]] * 32.042 / 0.791 + x[IX["E"]] * MW_FAME / RHO_FAME) / 1000.0     # L
        c_b = max(x[IX["Cat"]] + x[IX["OH"]], 1e-12) / V_bp
        out["pH"][k] = PH_REF + np.log10(c_b / C_B_REF)
    for k, sd in [("T", 0.05), ("T_J", 0.05), ("T_Jin", 0.05)]:
        out[k] = out[k] + rng.normal(0.0, sd, len(t))
    out["P_st"] = out["P_st"] + rng.normal(0.0, 0.02 * P0_W, len(t))
    # load cell: charge mass plus integrated methanol/water flows (kg)
    m0 = PILOT["m_oil_kg"] + x0[IX["A"]] * 32.042 / 1000.0 + x0[IX["Cat"]] * 138.205 / 1000.0
    flow = out["F_A"] * 32.042 / 1000.0 + out["F_W"] * 18.015 / 1000.0
    out["m_ld"] = m0 + np.concatenate([[0.0], np.cumsum(0.5 * (flow[1:] + flow[:-1]) * DT)])
    out["m_ld"] = out["m_ld"] + rng.normal(0.0, 0.2, len(t))
    out["pH"] = out["pH"] + rng.normal(0.0, 0.02, len(t))
    out["Q_J"] = FJ * p.rho_cp_J * (out["T_Jin"] - out["T_J"]) / 60.0 / 1000.0   # kW
    for k in ["T", "T_J", "T_Jin", "T_cmd"]:
        out[k] = out[k] - 273.15
    out["t"] = t
    p.a, p.eta = a_nom, 1.0
    out["a_rel"] = np.full(len(t), m if cls == "F3" else 1.0)
    out["dAV"] = np.full(len(t), m if cls == "F1" else 0.0)
    if return_states:
        out["_x0"], out["_p"] = x0, p
    return out
