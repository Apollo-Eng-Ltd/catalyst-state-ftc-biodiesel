"""Check casadi_plant against plant.rhs (random states) and the CVODES integrator against LSODA."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
from plant import rhs, IX as PIX, I_V, I_T, I_TJ   # noqa: E402
from pilot import pilot_case, PILOT                 # noqa: E402
from scipy.integrate import solve_ivp               # noqa: E402
import casadi_plant as cp                           # noqa: E402

rng = np.random.default_rng(0)
x0p, p = pilot_case("SWVO", group="veg"); p.isothermal = False
c = dict(rho_cp=p.rho_cp, UA=p.UA, V_J=p.V_J, rho_cp_J=p.rho_cp_J, F_J=PILOT["F_J_Lmin"])
f = cp.make_f(c)
worst = 0.0
for trial in range(200):
    xp = x0p.copy()
    for s in cp.SP:
        xp[PIX[s]] = x0p[PIX["TG"]] * rng.uniform(0, 3) * (0.05 if s in ("Cat", "OH", "FFA", "W", "Soap", "GL", "MG", "DG") else 1)
    xp[I_V] *= rng.uniform(0.95, 1.05); xp[I_T] = 273.15 + rng.uniform(50, 64); xp[I_TJ] = 273.15 + rng.uniform(25, 80)
    a_rel, eta, FW, kl, phi = rng.uniform(0.2, 1), rng.uniform(0.2, 1), rng.uniform(0, 5), rng.uniform(0, .02), rng.uniform(.1, 1)
    Tc, ucat, FA = 273.15 + rng.uniform(25, 80), rng.uniform(0, 2), rng.uniform(0, 20)
    a0 = p.a; p.a = a0 * a_rel; p.eta = eta
    Tin = xp[I_TJ] + phi * (Tc - xp[I_TJ])
    d = rhs(0.0, xp, p, lambda t: dict(T_Jin=Tin, F_J=c["F_J"], F_W=FW, F_A=FA - kl * xp[PIX["A"]], u_cat=ucat))
    p.a = a0; p.eta = 1.0
    x = np.r_[[xp[PIX[s]] for s in cp.SP], xp[I_V], xp[I_T], xp[I_TJ]]
    dc = np.array(f(x, [Tc, ucat, FA], [a_rel, eta, FW, kl, phi])).ravel()
    ref = np.r_[[d[PIX[s]] for s in cp.SP], d[I_V], d[I_T], d[I_TJ]]
    worst = max(worst, np.max(np.abs(dc - ref) / (np.abs(ref) + 1e-9 * np.abs(x) + 1e-12)))
print(f"max relative error casadi vs plant.rhs: {worst:.2e}")
assert worst < 1e-8
# integrator over 5 min from the pilot initial state
F = cp.make_integrator(c, 5.0)
x = np.r_[[x0p[PIX[s]] for s in cp.SP], x0p[I_V], x0p[I_T], x0p[I_TJ]]
xc = np.array(F(x0=x, p=[333.15, 0, 0, 1, 1, 0, 0, 1])["xf"]).ravel()
s = solve_ivp(rhs, (0, 5), x0p, args=(p, lambda t: dict(T_Jin=333.15, F_J=c["F_J"])), method="LSODA", rtol=1e-9, atol=1e-10)
xs = s.y[:, -1]; ref = np.r_[[xs[PIX[s_]] for s_ in cp.SP], xs[I_V], xs[I_T], xs[I_TJ]]
err = np.max(np.abs(xc - ref) / (np.abs(ref) + 1e-3 * x0p[PIX["TG"]]))   # floor: 0.1 % of TG0 (FFA, W sit at ~1e-7 mol)
print(f"CVODES vs LSODA after 5 min: max rel diff {err:.2e}")
assert err < 1e-5
print("casadi plant checks passed")
