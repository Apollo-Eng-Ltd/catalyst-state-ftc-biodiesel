"""Check torch_plant against plant.rhs at random states. Run: python3 test_torch_plant.py"""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
from plant import rhs, IX as PIX, I_V, I_T, I_TJ   # noqa: E402
from pilot import pilot_case                         # noqa: E402
import torch_plant as tp                             # noqa: E402

rng = np.random.default_rng(0)
x0, p = pilot_case("SWVO", group="veg")
p.isothermal = False
nTG0, V0 = x0[PIX["TG"]], x0[I_V]
worst_sp, worst_T = 0.0, 0.0
for trial in range(200):
    x = x0.copy()
    for s in tp.SP:
        x[PIX[s]] = abs(x0[PIX["TG"]] * rng.uniform(0, 3) * (0.05 if s in ("Cat", "OH", "FFA", "W", "Soap", "GL", "MG", "DG") else 1))
    x[I_V] = V0 * rng.uniform(0.95, 1.05); x[I_T] = 273.15 + rng.uniform(50, 64); x[I_TJ] = 273.15 + rng.uniform(25, 80)
    a_rel, eta, FW, kl = rng.uniform(0.2, 1), rng.uniform(0.2, 1), rng.uniform(0, 5), rng.uniform(0, 0.02)
    p.eta = eta; p_a = p.a; p.a = p_a * a_rel
    dx = rhs(0.0, x, p, lambda t: dict(T_Jin=x[I_TJ], F_J=0.0, F_W=FW, F_A=-kl * x[PIX["A"]]))
    p.a = p_a; p.eta = 1.0
    z = torch.tensor([x[PIX[s]] / nTG0 for s in tp.SP], dtype=torch.float64)
    dz, dzV, r = tp.dz_dt(z, torch.tensor(x[I_V] / V0, dtype=torch.float64), torch.tensor(x[I_T], dtype=torch.float64),
                          torch.tensor(a_rel, dtype=torch.float64), torch.tensor(eta, dtype=torch.float64),
                          torch.tensor(FW, dtype=torch.float64), torch.tensor(kl, dtype=torch.float64), nTG0, V0)
    ref = np.array([dx[PIX[s]] / nTG0 for s in tp.SP])
    worst_sp = max(worst_sp, np.max(np.abs(dz.numpy() - ref)) / (np.max(np.abs(ref)) + 1e-12))
    dT = tp.dT_dt(torch.tensor(x[I_T]), torch.tensor(x[I_TJ]), r, torch.tensor(x[I_V] / V0), V0, p.rho_cp, p.UA)
    worst_T = max(worst_T, abs(dT.item() - dx[I_T]) / (abs(dx[I_T]) + 1e-12))
print(f"max relative error: species {worst_sp:.2e}, temperature {worst_T:.2e}")
assert worst_sp < 1e-9 and worst_T < 1e-9
print("torch_plant matches plant.rhs")
