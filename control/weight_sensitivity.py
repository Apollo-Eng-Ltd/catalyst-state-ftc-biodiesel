"""Sensitivity of FT-NMPC (rule v2) to the economic weights of the NMPC objective.
Subset: first 6 scenarios per class of the fresh Monte Carlo (48 batches); one weight changed at a time
(catalyst w_cat 5 -> 1 and 25; methanol w_A 1 -> 0.2 and 5). Base case taken from mc2_results.jsonl.
Output: wsens_results.jsonl
"""
import os, sys, json, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from scenarios import mc_cases
from controllers2 import run_nmpc2, W_DEF
OUTF = os.path.join(HERE, "wsens_results.jsonl")
VARIANTS = {"w_cat=1": dict(cat=1.0), "w_cat=25": dict(cat=25.0), "w_A=0.2": dict(A=0.2), "w_A=5": dict(A=5.0)}


def one(a):
    i, c, name = a
    w = dict(W_DEF, **VARIANTS[name]); t0 = time.time()
    try:
        r, _ = run_nmpc2(c, est="gru", rule="v2", seed=c["seed"], w=w); r.pop("decisions", None); err = None
    except Exception as e:
        r, err = {}, repr(e)[:200]
    r.update(dict(case=i, variant=name, cls=c["cls"], wall=round(time.time() - t0, 1), error=err))
    return r


if __name__ == "__main__":
    cases = mc_cases(30, seed=77)
    idx = [ci * 30 + j for ci in range(8) for j in range(6)]
    done = set()
    if os.path.exists(OUTF):
        for l in open(OUTF):
            d = json.loads(l); done.add((d["case"], d["variant"]))
    jobs = [(i, cases[i], v) for i in idx for v in VARIANTS if (i, v) not in done]
    with Pool(2) as p, open(OUTF, "a") as f:
        for r in p.imap_unordered(one, jobs):
            f.write(json.dumps(r, default=float) + "\n"); f.flush()
    print("done")
