"""Fresh Monte Carlo (seed 77, 240 scenarios) re-run for FT-NMPC v2 and PID-35 with side-reaction heats in
the PLANT only (dH_rN = dH_rSE = -57.3 kJ/mol; see side_heat_check.py). Resumable: mc2_sideheat.jsonl"""
import os, sys, json, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import pilot, plant_sim
_orig = pilot.pilot_case
def patched(*a, **k):
    x0, p = _orig(*a, **k); p.dH = dict(p.dH, rN=-57.3e3, rSE=-57.3e3); return x0, p
plant_sim.pilot_case = patched
from scenarios import mc_cases
from controllers import run_pid
from controllers2 import run_nmpc2
OUTF = os.path.join(HERE, "mc2_sideheat.jsonl")
def one(a):
    i, c, ctrl = a; t0 = time.time()
    try:
        r, _ = run_pid(c, 35.0, seed=c["seed"]) if ctrl == "PID-35" else run_nmpc2(c, est="gru", rule="v2", seed=c["seed"])
        r.pop("decisions", None); err = None
    except Exception as e:
        r, err = {}, repr(e)[:200]
    r.update(dict(case=i, ctrl=ctrl, cls=c["cls"], wall=round(time.time() - t0, 1), error=err)); return r
if __name__ == "__main__":
    cases = mc_cases(30, seed=77); done = set()
    if os.path.exists(OUTF):
        for l in open(OUTF):
            d = json.loads(l); done.add((d["case"], d["ctrl"]))
    jobs = [(i, c, k) for i, c in sorted(enumerate(cases), key=lambda ic: (ic[0] % 30, ic[0] // 30)) for k in ["PID-35", "FT-NMPC-v2"] if (i, k) not in done]
    with Pool(2) as p, open(OUTF, "a") as f:
        for r in p.imap_unordered(one, jobs):
            f.write(json.dumps(r, default=float) + "\n"); f.flush()
    print("done")
