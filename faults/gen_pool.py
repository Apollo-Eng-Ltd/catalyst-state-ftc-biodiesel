"""Unlabelled new-feed pool for adaptation: 80 Jatropha charges, seeds disjoint from test_jco.
Labels are saved only so that the adaptation run can REPORT validation error; training never uses them."""
import os, numpy as np, pandas as pd
from multiprocessing import Pool
from generate_dataset import jco_cases, run, OUT, MEAS, TRUTH, STATE_NAMES, FPAR, DT
cases = jco_cases(80, seed=14)
for c in cases: c["split"] = "pool_jco"
with Pool(os.cpu_count()) as pool:
    res = pool.map(run, cases, chunksize=4)
ok = [(c, a) for c, a, e in res if e is None]
X = np.stack([a[0] for _, a in ok]); S = np.stack([a[1] for _, a in ok])
rows = []
for c, _ in ok:
    r = {k: v for k, v in c.items() if not k.startswith("_") and k != "mismatch"}
    r.update({f"mm_{k}": v for k, v in c["mismatch"].items()}); rows.append(r)
meta = pd.DataFrame(rows); meta.insert(0, "batch_id", [f"pool_jco_{i:05d}" for i in range(len(meta))])
meta.to_csv(f"{OUT}/pool_jco_meta.csv", index=False)
np.savez_compressed(f"{OUT}/pool_jco_X.npz", X=X, channels=np.array(MEAS + TRUTH), dt=DT)
np.savez_compressed(f"{OUT}/pool_jco_S.npz", S=S, channels=np.array(STATE_NAMES + FPAR), dt=DT)
print("pool_jco", X.shape, "failed", len(res) - len(ok))
