"""Monte Carlo comparison on identical scenarios.
Controllers: PID-35 (tuned batch time), PID-60 (conventional 1-h batch), NMPC-nominal, FT-NMPC.
Results appended per batch to control/mc_results.jsonl (resumable); summary in mc_summary.json.
Usage: python3 monte_carlo.py <n_per_class>
"""
import os, sys, json, time, datetime
import numpy as np, pandas as pd
from multiprocessing import Pool
from scenarios import mc_cases
from controllers import run_pid, run_nmpc

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "mc_results.jsonl")
T_PID = json.load(open(os.path.join(HERE, "pid_tstop.json")))["t_stop"]
# Stop-rule back-off = the closed-loop estimator's validation RMSE of X_E on held-out interpolation
# batches (0.72 points, estimator/est_data_cl evaluation); chosen from the estimator, not from MC results.
BACKOFF = 0.0072
CTRLS = ["PID-35", "PID-60", "NMPC-nom", "FT-NMPC", "NMPC-nom-b", "FT-NMPC-b"]


def one(args):
    i, c, ctrl = args
    t0 = time.time()
    try:
        if ctrl == "PID-35":
            r, _ = run_pid(c, T_PID, seed=c["seed"])
        elif ctrl == "PID-60":
            r, _ = run_pid(c, 60.0, seed=c["seed"])
        else:
            bo = BACKOFF if ctrl.endswith("-b") else 0.0
            r, _ = run_nmpc(c, "nominal" if ctrl.startswith("NMPC-nom") else "ft", seed=c["seed"], backoff=bo)
        r.pop("decisions", None)
        err = None
    except Exception as e:
        r, err = {}, str(e)[:200]
    r.update(dict(case=i, ctrl=ctrl, cls=c["cls"], s=c["s"], t_f=c["t_f"], wall=round(time.time() - t0, 1), error=err))
    return r


def main(n):
    cases = mc_cases(n)
    done = set()
    if os.path.exists(OUTF):
        for line in open(OUTF):
            d = json.loads(line); done.add((d["case"], d["ctrl"]))
    jobs = [(i, c, k) for i, c in enumerate(cases) for k in CTRLS if (i, k) not in done]
    print("jobs", len(jobs), flush=True)
    with Pool(2) as pool, open(OUTF, "a") as f:
        for r in pool.imap_unordered(one, jobs):
            f.write(json.dumps(r, default=float) + "\n"); f.flush()
    df = pd.read_json(OUTF, lines=True)
    df = df[df.case < len(cases)]
    summ = {}
    for k, g in df.groupby("ctrl"):
        summ[k] = dict(n=len(g), errors=int(g.error.notna().sum()), on_spec=round(float(g.on_spec.mean()), 3),
                       XE_mean=round(100 * float(g.XE_stop.mean()), 2), t_stop_mean=round(float(g.t_stop.mean()), 1),
                       cat_topup_kg=round(float(g.cat_topup_kg.mean()), 3), meoh_kg=round(float(g.meoh_makeup_kg.mean()), 2),
                       T_viol=int(g.T_viol.sum()), IAE=round(float(g.IAE.mean()), 1), ISE=round(float(g.ISE.mean()), 1),
                       ITAE=round(float(g.ITAE.mean()), 0),
                       solve_mean=round(float(g.solve_mean.mean()), 2) if "solve_mean" in g and g.solve_mean.notna().any() else None,
                       solve_max=round(float(g.solve_max.max()), 2) if "solve_max" in g and g.solve_max.notna().any() else None,
                       on_spec_by_class=g.groupby("cls").on_spec.mean().round(3).to_dict())
    json.dump(dict(run_utc=datetime.datetime.now(datetime.UTC).isoformat(), n_per_class=n, t_pid=T_PID, backoff=BACKOFF, summary=summ),
              open(os.path.join(HERE, "mc_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
