"""Summary and paired statistics for the fresh Monte Carlo (mc2_results.jsonl).
McNemar exact test on the on-spec indicator (paired by scenario), bootstrap 95 % CI of the on-spec rate,
per-class on-spec table. Output: mc2_summary.json"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_json(os.path.join(HERE, "mc2_results.jsonl"), lines=True)
err = df[df.error.notna()]
df = df[df.error.isna()]
full = df.groupby("case").ctrl.nunique(); keep = full[full == df.ctrl.nunique()].index
df = df[df.case.isin(keep)]
O = df.pivot(index="case", columns="ctrl", values="on_spec").astype(bool)
cls = df.groupby("case").cls.first()
rng = np.random.default_rng(0)
out = dict(n_scenarios=int(len(O)), errors=int(len(err)), summary={}, mcnemar={}, by_class={})
for k, g in df.groupby("ctrl"):
    o = O[k].values.astype(float); bs = o[rng.integers(0, len(o), (4000, len(o)))].mean(1)
    out["summary"][k] = dict(on_spec=float(o.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                             XE_mean=float(100 * g.XE_stop.mean()), t_stop=float(g.t_stop.mean()),
                             meoh_kg=float(g.meoh_makeup_kg.mean()), cat_kg=float(g.cat_topup_kg.mean()),
                             T_viol=int(g.T_viol.sum()),
                             solve_mean=float(g.solve_mean.mean()) if g.solve_mean.notna().any() else None,
                             solve_max=float(g.solve_max.max()) if g.solve_max.notna().any() else None)
for a, b in [("FT-NMPC-v2", "FT-NMPC-v1"), ("FT-NMPC-v2", "NMPC-nom"), ("FT-NMPC-v2", "PID-35"), ("FT-NMPC-v2", "PID-60"),
             ("FT-NMPC-v2", "EKF-NMPC-v2"), ("FT-NMPC-v1", "NMPC-nom"), ("EKF-NMPC-v2", "NMPC-nom"), ("EKF-NMPC-v2", "PID-35")]:
    better = int((O[a] & ~O[b]).sum()); worse = int((~O[a] & O[b]).sum())
    p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
    out["mcnemar"][f"{a} vs {b}"] = dict(better=better, worse=worse, p=float(p))
for c in sorted(cls.unique()):
    out["by_class"][c] = {k: float(O.loc[cls == c, k].mean()) for k in O.columns}
json.dump(out, open(os.path.join(HERE, "mc2_summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
