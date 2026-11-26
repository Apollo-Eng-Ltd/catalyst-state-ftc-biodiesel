"""Fresh Monte Carlo (new seeds, never used in design) for the stop-rule fix and the EKF baseline.
240 scenarios = 30 per class from scenarios.mc_cases(30, seed=77); the first study used seed 21 and served as the
development set for stop rule v2. Results per batch appended to mc2_results.jsonl (resumable).
Usage: python3 monte_carlo2.py [n_per_class]
"""
import os, sys, json, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from scenarios import mc_cases
from controllers import run_pid, run_nmpc
from controllers2 import run_nmpc2

OUTF = os.path.join(HERE, "mc2_results.jsonl")
T_PID = json.load(open(os.path.join(HERE, "pid_tstop.json")))["t_stop"]
CTRLS = ["PID-35", "PID-60", "NMPC-nom", "FT-NMPC-v1", "FT-NMPC-v2", "EKF-NMPC-v2"]
SEED = 77


def one(args):
    i, c, ctrl = args
    t0 = time.time()
    try:
        if ctrl == "PID-35":
            r, _ = run_pid(c, T_PID, seed=c["seed"])
        elif ctrl == "PID-60":
            r, _ = run_pid(c, 60.0, seed=c["seed"])
        elif ctrl == "NMPC-nom":
            r, _ = run_nmpc(c, "nominal", seed=c["seed"])
        elif ctrl == "FT-NMPC-v1":
            r, _ = run_nmpc(c, "ft", seed=c["seed"])
        elif ctrl == "FT-NMPC-v2":
            r, _ = run_nmpc2(c, est="gru", rule="v2", seed=c["seed"])
        elif ctrl == "EKF-NMPC-v2":
            r, _ = run_nmpc2(c, est="ekf", rule="v2", seed=c["seed"])
        r.pop("decisions", None)
        err = None
    except Exception as e:
        r, err = {}, repr(e)[:200]
    r.update(dict(case=i, ctrl=ctrl, cls=c["cls"], s=c["s"], t_f=c["t_f"], oil=c["oil"], wall=round(time.time() - t0, 1), error=err))
    return r


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cases = mc_cases(n, seed=SEED)
    done = set()
    if os.path.exists(OUTF):
        for line in open(OUTF):
            d = json.loads(line); done.add((d["case"], d["ctrl"]))
    jobs = [(i, c, k) for i, c in sorted(enumerate(cases), key=lambda ic: (ic[0] % n, ic[0] // n)) for k in CTRLS if (i, k) not in done]  # classes interleaved
    print("jobs", len(jobs), flush=True)
    with Pool(2) as pool, open(OUTF, "a") as f:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            f.write(json.dumps(r, default=float) + "\n"); f.flush()
    print("done", flush=True)
