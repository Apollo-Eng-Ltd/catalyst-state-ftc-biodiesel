"""Python references for the Simulink headline scenarios and the data file for MATLAB.
Writes ftc_sup_data.json (estimator weights, scaling, noise sequence, cases) and refs_headline.json."""
import os, sys, json, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ["model", "control", "estimator", "faults"]:
    sys.path.insert(0, os.path.join(HERE, "..", d))
from controllers import run_pid, run_nmpc, estimate
from faults import magnitude

SEED = 7
BASE = dict(oil="SWVO", cat_wt=4.0, ratio=6.0, T0_C=53.0, water_wt=0.0002, AV=1.088, Tsp_C=60.0, group="veg", seed=SEED)
CASES = {
    "normal":        dict(BASE, cls="N"),
    "degraded_oil":  dict(BASE, cls="F3", s=0.6),
    "water_ingress": dict(BASE, cls="F2", s=0.5, t_f=10.0, shape="abrupt"),
    "methanol_loss": dict(BASE, cls="E3", s=0.5, t_f=10.0, shape="abrupt"),
}


def traj(b, res):
    A = b.arrays()
    out = dict(res)
    out.pop("decisions", None)
    out.update(t=A["t"][::4].tolist(), X_E=A["X_E"][::4].tolist(), T=A["T_true"][::4].tolist(),
               T_cmd=A["T_cmd"][::4].tolist())
    if "decisions" in res:
        out["dec"] = [dict(t=d["t"], XE_est=d["XE_est"], T_cmd=d["T_cmd"], u_cat=d["u_cat"], F_A=d["F_A"], q=d["q"])
                      for d in res["decisions"]]
    return out


if __name__ == "__main__":
    refs = {}
    for name, c in CASES.items():
        refs[name] = {}
        r, b = run_pid(c, 35.0, seed=SEED); refs[name]["PID35"] = traj(b, r)
        for mode, tag in [("nominal", "NMPC"), ("ft", "FTNMPC")]:
            t0 = time.time(); r, b = run_nmpc(c, mode, seed=SEED); refs[name][tag] = traj(b, r)
            print(name, tag, round(r["XE_stop"], 4), r["t_stop"], round(time.time() - t0, 1), flush=True)
        print(name, "PID35", round(refs[name]["PID35"]["XE_stop"], 4), flush=True)
    json.dump(dict(cases=CASES, refs=refs), open(os.path.join(HERE, "refs_headline.json"), "w"))
    # data for MATLAB
    ck = torch.load(os.path.join(HERE, "..", "estimator", "est_data_cl.pt"), weights_only=False)
    W = {k.replace(".", "_"): v.double().numpy().tolist() for k, v in ck["state"].items()}
    noise = np.random.default_rng(SEED).normal(size=(481 * 6,)).reshape(481, 6).tolist()
    mag = {n: magnitude(c["cls"], c.get("s", 0.0)) for n, c in CASES.items()}
    json.dump(dict(W=W, mu_x=np.asarray(ck["mu_x"]).tolist(), sd_x=np.asarray(ck["sd_x"]).tolist(),
                   mu_y=np.asarray(ck["mu_y"]).tolist(), sd_y=np.asarray(ck["sd_y"]).tolist(), noise=noise,
                   mag=mag), open(os.path.join(HERE, "ftc_sup_data.json"), "w"))
    print("done")
