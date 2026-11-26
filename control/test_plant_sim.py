"""plant_sim.Batch under PI must reproduce faults.simulate_batch (same case, noise-free comparison of truth)."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "faults"))
from faults import simulate_batch, CLASSES   # noqa: E402
from plant_sim import Batch                  # noqa: E402

worst = 0.0
for cls in CLASSES:
    case = dict(oil="SWVO", ratio=6.5, cat_wt=4.2, T0_C=53, Tsp_C=59, cls=cls, s=0.6, t_f=22.0, shape="ramp",
                tau_r=15.0, sign=-1.0, mismatch=dict(a=0.9, k_SE=0.012, k_H=1.1, c_th=0.95, UA=1.05))
    ref = simulate_batch(case, seed=3)
    b = Batch(case, seed=3)
    for k in range(24):
        b.step(dict(mode="pi", T_sp=273.15 + 59.0))
    A = b.arrays()
    dX = np.max(np.abs(A["X_E"] - ref["X_E"])); dT = np.max(np.abs(A["T_true"] - (ref["state"][:, 12] - 273.15)))
    worst = max(worst, dX * 100, dT)
    print(f"{cls}: max |dX_E| {100 * dX:.2e} pp, max |dT| {dT:.2e} K")
assert worst < 0.05, worst
print("plant_sim reproduces faults.simulate_batch")
