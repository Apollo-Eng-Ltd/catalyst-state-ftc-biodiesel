"""Effect of side-reaction heats left out of the plant energy balance (dH_rN = dH_rSE = 0 in pilot.py).
Bound: ester saponification and fatty-acid neutralisation each release about 57 kJ/mol (alkaline hydrolysis
of poly(vinyl acetate) ester groups, -57.3 kJ/mol; neutralisation enthalpy of the acid formed, -55.8 kJ/mol;
Wadso and Karlsson, Polym Degrad Stab 98 (2013) 73-78). rW and rH are proton transfers between methoxide and
hydroxide (taken as thermoneutral; worst case also -57.3 kJ/mol). The heats are added to the PLANT only; the
NMPC model and the estimator stay unchanged (an extra plant-model mismatch). Output: side_heat_check.json"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "model")); sys.path.insert(0, os.path.join(HERE, "..", "matlab", "reference"))
import pilot, plant_sim
import controllers, controllers2
from controllers import run_pid
from controllers2 import run_nmpc2
import make_refs as M
_orig = pilot.pilot_case
MODE = {"dH": None}


def patched(*a, **k):
    x0, p = _orig(*a, **k)
    if MODE["dH"] is not None:
        p.dH = dict(p.dH, **MODE["dH"])
    return x0, p


plant_sim.pilot_case = patched          # plant only (Batch); controllers keep their own nominal model
VARIANTS = {"none (as run)": None, "rN,rSE -57.3": dict(rN=-57.3e3, rSE=-57.3e3),
            "all side -57.3 (worst)": dict(rN=-57.3e3, rSE=-57.3e3, rW=-57.3e3, rH=-57.3e3)}
out = {}
for cname, c in M.CASES.items():
    for vname, dh in VARIANTS.items():
        MODE["dH"] = dh
        rp, bp = run_pid(c, 35.0, seed=M.SEED)
        rf, bf = run_nmpc2(c, est="gru", rule="v2", seed=M.SEED)
        out[f"{cname} | {vname}"] = dict(PID35_XE=rp["XE_stop"], PID35_Tpeak=rp["T_peak"], FT_XE=rf["XE_stop"], FT_t=rf["t_stop"],
                                         FT_Tpeak=rf["T_peak"], FT_onspec=rf["on_spec"])
        print(cname, "|", vname, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[f"{cname} | {vname}"].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "side_heat_check.json"), "w"), indent=1)
