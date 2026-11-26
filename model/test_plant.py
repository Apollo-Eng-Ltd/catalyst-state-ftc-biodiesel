"""Verification of plant.py. Run: python3 test_plant.py"""
import json, numpy as np
from plant import *

out = {}

# 1. Noureddini-Zhu limit: single phase, constant catalyst at C_REF, no side reactions, 50 C, 90 min
p = Params(a=1.0, GL_sat=1e6, k_N=0.0, k_S=0.0, k_SE=0.0)
x0 = initial_state(m_oil_g=200, MW_oil=872, ratio=6, cat_g=0, AV=0, water_wt=0, T_C=50)
x0[IX["Cat"]] = C_REF * x0[I_V]
s = simulate(x0, p, 90)
xe = X_E(s.y[:, -1], x0)
out["NZ_limit_XE_90min"] = round(xe, 4)
assert abs(xe - 0.838) < 2e-3, xe          # independent check: 0.838; Fig. 2 of NZ ~0.85

# 2. Invariants: all reactions, non-isothermal, catalyst top-up and methanol dosing
p = Params(a=0.3, GL_sat=0.05, k_N=1e3, k_S=5.0, k_SE=0.5, k_H=0.01, Ea_SE=6e4, K_cg=2.0, c_th=0.03, isothermal=False)
x0 = initial_state(ratio=6, cat_g=4, AV=2.6, water_wt=0.002, T_C=55)
u = lambda t: dict(u_cat=2e-5 if 20 < t < 30 else 0.0, F_A=5e-4 if t < 40 else 0.0,
                   T_Jin=336.0, F_J=0.2, Q_h=50.0)
s = simulate(x0, p, 120, u=u)
inv = invariants(s.y[:, -1], x0)
out["invariants_full_model"] = {k: float(f"{v:.2e}") for k, v in inv.items()}
assert all(abs(v) < 1e-11 for v in inv.values()), inv
assert s.success and np.all(s.y[:len(SP), :] > -1e-12)

# 3. Glycerol phase lifts the equilibrium cap (1:6, 60 C, constant catalyst, long time)
lift = {}
for gl_sat in [1e6, 0.5, 0.1, 0.05, 0.02]:
    p = Params(a=1.0, GL_sat=gl_sat, k_N=0.0, k_S=0.0, k_SE=0.0)
    x0 = initial_state(ratio=6, cat_g=0, AV=0, water_wt=0, T_C=60)
    x0[IX["Cat"]] = C_REF * x0[I_V]
    lift[str(gl_sat)] = round(X_E(simulate(x0, p, 3000).y[:, -1], x0), 4)
out["equilibrium_XE_vs_GLsat_1to6_60C"] = lift
assert lift["1000000.0"] < 0.86 and lift["0.02"] > 0.965

# 4. Thermal: no reaction, steady state equals the analytic jacket/reactor solution
p = Params(a=0.0, k_N=0, k_S=0, isothermal=False)
x0 = initial_state(cat_g=0, AV=0, water_wt=0, T_C=25)
F_J, TJin, Qh = 0.5, 340.0, 0.0
s = simulate(x0, p, 600, u=lambda t: dict(F_J=F_J, T_Jin=TJin, Q_h=Qh))
out["thermal_T_end_K"] = round(s.y[I_T, -1], 4)
assert abs(s.y[I_T, -1] - TJin) < 1e-3       # with no losses, reactor tends to jacket inlet

# 5. Catalyst budget: FFA neutralisation consumes base 1:1 and fast
p = Params(a=1.0, k_N=1e3, k_S=0, k_SE=0)
x0 = initial_state(cat_g=4, AV=2.61, water_wt=0, T_C=60)
s = simulate(x0, p, 5)
used = x0[IX["Cat"]] - s.y[IX["Cat"], -1]
out["JCO_4wt_base_used_by_FFA_frac"] = round(used / x0[IX["Cat"]], 4)
assert abs(used - x0[IX["FFA"]]) < 1e-6

print(json.dumps(out, indent=1))
json.dump(out, open("test_plant.run.json", "w"), indent=1)
print("all checks passed")
