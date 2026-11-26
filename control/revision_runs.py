"""Additional closed-loop runs on the seed-77 test scenarios.

FT-v2-log      FT-NMPC, rule v2, with the estimate and the true conversion logged at every decision
               (empirical coverage of the stop margins on the test scenarios)
Oracle-v2      NMPC and rule v2 with the true state and fault parameters, margins zero
FT-v2-batch    FT-NMPC, rule v2, with batch-level conformal margins (batch_calibration.py)
thr:*          rule v2 with one stop threshold changed, on the first 6 scenarios per class
Output: revision_results.jsonl (resumable)
"""
import os, sys, json, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from scenarios import mc_cases
from controllers2 import run_nmpc2

OUTF = os.path.join(HERE, "revision_results.jsonl")
CAL_B = json.load(open(os.path.join(HERE, "stoprule_calibration_batch.json")))
THR = {"thr:g1x2": dict(g1=2e-3), "thr:g1x0.5": dict(g1=5e-4), "thr:g2x2": dict(g2=1e-3, rise=1e-3),
       "thr:noconfirm": dict(confirm=False)}


def one(a):
    i, c, name = a
    t0 = time.time()
    try:
        if name == "FT-v2-log":
            r, _ = run_nmpc2(c, est="gru", rule="v2", seed=c["seed"])
            r["dec"] = [[d["t"], float(d["XE_est"]), d["XE_true"], bool(d["alarm"])] for d in r["decisions"]]
        elif name == "Oracle-v2":
            r, _ = run_nmpc2(c, est="oracle", rule="v2", seed=c["seed"], cal_override=dict(delta0=0.0, delta1=0.0))
        elif name == "FT-v2-batch":
            r, _ = run_nmpc2(c, est="gru", rule="v2", seed=c["seed"], cal_override=CAL_B)
        else:
            r, _ = run_nmpc2(c, est="gru", rule="v2", seed=c["seed"], thr=THR[name])
        r.pop("decisions", None); err = None
    except Exception as e:
        r, err = {}, repr(e)[:200]
    r.update(dict(case=i, ctrl=name, cls=c["cls"], s=c["s"], t_f=c["t_f"], wall=round(time.time() - t0, 1), error=err))
    return r


if __name__ == "__main__":
    cases = mc_cases(30, seed=77)
    sub = [ci * 30 + j for ci in range(8) for j in range(6)]
    order = sorted(range(240), key=lambda i: (i % 30, i // 30))
    jobs = [(i, cases[i], k) for i in order for k in ("FT-v2-log", "Oracle-v2")]
    jobs += [(i, cases[i], k) for i in sub for k in THR]
    jobs += [(i, cases[i], "FT-v2-batch") for i in order]
    done = set()
    if os.path.exists(OUTF):
        for l in open(OUTF):
            d = json.loads(l); done.add((d["case"], d["ctrl"]))
    jobs = [j for j in jobs if (j[0], j[2]) not in done]
    print("jobs", len(jobs), flush=True)
    with Pool(2) as p, open(OUTF, "a") as f:
        for r in p.imap_unordered(one, jobs, chunksize=1):
            f.write(json.dumps(r, default=float) + "\n"); f.flush()
    print("done", flush=True)
