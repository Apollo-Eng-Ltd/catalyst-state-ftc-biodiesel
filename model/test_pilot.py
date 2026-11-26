"""Checks on the pilot reactor. Run: python3 test_pilot.py"""
import json, numpy as np
from scipy.integrate import solve_ivp
from plant import rhs, rates, X_E, IX, I_T, I_TJ, I_V, invariants, NU
from pilot import pilot_case, PILOT, geometry

out = {}
x0, p = pilot_case("SWVO", T0_C=25.0)
H, A = geometry(x0[I_V])
out["charge"] = dict(V_L=round(x0[I_V], 1), H_m=round(H, 3), A_m2=round(A, 2),
                     UA_W_K=round(p.UA / 60, 0), V_J_L=round(p.V_J, 1), rho_cp_J_LK=round(p.rho_cp, 0))

# 1. Heat-up 25 -> 60 C with full heater, no reaction (a = 0): energy balance closes
p0 = pilot_case("SWVO", T0_C=25.0, extra=dict(a=0.0, k_N=0.0, k_S=0.0, k_H=0.0))[1]
Qh = PILOT["Q_h_max_W"] * 60.0
u = lambda t: dict(Q_h=Qh, F_J=PILOT["F_J_Lmin"])
ev = lambda t, x, p, u: x[I_T] - 333.15
ev.terminal = True
s = solve_ivp(rhs, (0, 600), x0, args=(p0, u), method="LSODA", rtol=1e-8, atol=1e-10, events=ev, dense_output=True)
t60 = s.t_events[0][0]
E_in = Qh * t60
E_store = p0.rho_cp * x0[I_V] * (s.y[I_T, -1] - x0[I_T]) + p0.rho_cp_J * p0.V_J * (s.y[I_TJ, -1] - x0[I_TJ])
out["heatup_25_to_60C_min"] = round(t60, 1)
out["heatup_energy_closure_rel"] = float(f"{(E_in - E_store) / E_in:.2e}")
assert abs(E_in - E_store) / E_in < 1e-5

# 2. Nominal batch: starts at 55 C, PI on the TCU jacket-inlet temperature, setpoint 60 C, 120 min
x0, p = pilot_case("SWVO", T0_C=55.0)
FJ, Kc, tauI, Tsp = PILOT["F_J_Lmin"], 6.0, 8.0, 333.15
lo, hi = PILOT["T_Jin_min_K"], PILOT["T_Jin_max_K"]
def tjin(x, e_int):
    return Tsp + Kc * ((Tsp - x[I_T]) + e_int / tauI)
def rhs_pi(t, z):
    x, e_int = z[:-1], z[-1]
    raw = tjin(x, e_int)
    Tj = np.clip(raw, lo, hi)
    dx = rhs(t, x, p, lambda tt: dict(T_Jin=Tj, F_J=FJ))
    return np.append(dx, (Tsp - x[I_T]) + (Tj - raw) / Kc)          # back-calculation anti-windup
s = solve_ivp(rhs_pi, (0, 120), np.append(x0, 0.0), method="LSODA", rtol=1e-7, atol=1e-10, dense_output=True)
t = np.linspace(0, 120, 241)
X = np.array([100 * X_E(s.sol(tt)[:-1], x0) for tt in t])
T = np.array([s.sol(tt)[I_T] - 273.15 for tt in t])
TJ = np.array([np.clip(tjin(s.sol(tt)[:-1], s.sol(tt)[-1]), lo, hi) - 273.15 for tt in t])
out["nominal_TJin_range_C"] = [round(TJ.min(), 1), round(TJ.max(), 1)]
i = int(np.argmax(X))
on = t[X >= 96.5]
out["nominal_SWVO"] = dict(peak_XE=round(X[i], 2), t_peak=float(t[i]),
                           on_spec_window_min=[float(on[0]), float(on[-1])] if len(on) else None,
                           T_max_C=round(T.max(), 2), T_end_C=round(T[-1], 2))
assert T.max() < PILOT["T_max_K"] - 273.15
inv = invariants(s.y[:-1, -1], x0)
out["invariants_pilot"] = {k: float(f"{v:.1e}") for k, v in inv.items()}
assert all(abs(v) < 1e-6 * x0[IX["TG"]] for v in inv.values())

# 3. Side-reaction heat is negligible: worst case (JCO, AV 6, water 0.3 wt%) with a -57 kJ/mol bound on each
x0w, pw = pilot_case("JCO", AV=6.0, water_wt=0.003, T0_C=60.0, extra=dict(isothermal=True))
sw = solve_ivp(rhs, (0, 120), x0w, args=(pw, lambda t: {}), method="LSODA", rtol=1e-7, atol=1e-10)
xe = sw.y[:, -1]
n_ester = xe[IX["E"]] + xe[IX["Soap"]] - x0w[IX["FFA"]]      # esters formed (incl. later saponified)
n_side = xe[IX["Soap"]] + (xe[IX["OH"]])                        # moles through rN, rW/rH, rSE
out["side_heat_crude_upper_bound_worst_case"] = round(57e3 * n_side / (9.8e3 * n_ester), 3)

print(json.dumps(out, indent=1))
json.dump(out, open("test_pilot.run.json", "w"), indent=1)
print("pilot checks passed")
