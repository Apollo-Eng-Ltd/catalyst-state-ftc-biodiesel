"""Test scenarios. Monte Carlo cases share recipes, hidden mismatch and faults across controllers.
Faults: all eight classes, severity s ~ U(0, 1) (whole range: controllers are not trained on it),
onset t_f ~ U(5, 30) min, i.e. before the fault-free peak (~31 min), where faults decide whether the
batch reaches specification. Seeds are disjoint from the estimator and classifier datasets.
"""
import os, sys
import numpy as np
from scipy.stats import qmc
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "faults")); sys.path.insert(0, os.path.join(HERE, "..", "model"))
from generate_dataset import mismatch, OILS          # noqa: E402
from faults import CLASSES                            # noqa: E402
from pilot import AV_2013                             # noqa: E402


def mc_cases(n_per_class, seed=21, classes=CLASSES):
    cases = []
    for ci, cls in enumerate(classes):
        U = qmc.LatinHypercube(d=16, seed=seed + 101 * ci).random(n_per_class)
        for j, u in enumerate(U):
            oil = OILS[int(u[0] * len(OILS))]
            c = dict(cls=cls, oil=oil, cat_wt=3.5 + 1.5 * u[1], ratio=6 + 2 * u[2], T0_C=50 + 5 * u[3],
                     Tsp_C=60.0, water_wt=1e-4 + 2e-4 * u[5], AV=AV_2013[oil] * (0.8 + 0.4 * u[6]),
                     s=0.0, t_f=0.0, shape="abrupt", tau_r=20.0, sign=1.0, seed=int(seed * 1000 + ci * 100 + j),
                     mismatch=mismatch(u[11:16]))
            if cls != "N":
                c["s"] = float(u[7])
                if cls in ("F2", "E1", "E2", "E3", "E4"):
                    c["t_f"] = 5 + 25 * float(u[8]); c["shape"] = "abrupt" if u[9] < 0.5 else "ramp"
                    c["tau_r"] = 10 + 20 * float(u[10])
                if cls == "E4":
                    c["sign"] = 1.0 if (u[9] * 7) % 1 < 0.5 else -1.0
            cases.append(c)
    return cases
