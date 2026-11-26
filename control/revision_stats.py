"""Summary of revision_results.jsonl: coverage of the stop margins on the test scenarios, oracle NMPC,
batch-level margins and stop-threshold sensitivity. Output: revision_summary.json"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from scenarios import mc_cases
from economics import table, BASE, boot_diff

cases = mc_cases(30, seed=77)
R = pd.read_json(os.path.join(HERE, "revision_results.jsonl"), lines=True)
R = R[R.error.isna()].copy(); R["oil"] = [cases[i]["oil"] for i in R.case]
M = pd.read_json(os.path.join(HERE, "mc2_results.jsonl"), lines=True); M = M[M.error.isna()]
CAL = json.load(open(os.path.join(HERE, "stoprule_calibration.json")))
CALB = json.load(open(os.path.join(HERE, "stoprule_calibration_batch.json")))
out = {}

# coverage on the test scenarios (FT-NMPC rule v2, decisions every 5 min)
L = R[R.ctrl == "FT-v2-log"]
s0, s1, b0, b1 = [], [], [], []
for _, r in L.iterrows():
    e0, e1 = [], []
    for t, xh, xt, al in r.dec:
        if t >= 10 and 0.955 < xh < 0.99:
            (e1 if al else e0).append(xh - xt)
    s0 += e0; s1 += e1
    if e0: b0.append(max(e0))
    if e1: b1.append(max(e1))
s0, s1, b0, b1 = map(np.array, (s0, s1, b0, b1))
out["coverage"] = dict(
    n_batches=int(len(L)), n_dec_no_alarm=int(len(s0)), n_dec_alarm=int(len(s1)),
    sample_cov_delta0=float((s0 <= CAL["delta0"]).mean()), sample_cov_delta1=float((s1 <= max(CAL["delta0"], CAL["delta1"])).mean()),
    batch_cov_delta0=float((b0 <= CAL["delta0"]).mean()), batch_cov_delta1=float((b1 <= max(CAL["delta0"], CAL["delta1"])).mean()),
    batch_cov_batchdelta0=float((b0 <= CALB["delta0"]).mean()), batch_cov_batchdelta1=float((b1 <= CALB["delta1"]).mean()),
    err_sd_pp=float(100 * np.r_[s0, s1].std()), err_mean_pp=float(100 * np.r_[s0, s1].mean()),
    n_batches_with_dec=int(len(b0) + len(b1)))
# overestimation at the stop: stopped believing on spec while off spec
L2 = L.copy(); L2["xh_stop"] = [r[-1][1] for r in L.dec]
out["coverage"]["stopped_offspec"] = int((~L2.on_spec).sum())
out["coverage"]["stopped_offspec_est_above_spec"] = int(((~L2.on_spec) & (L2.xh_stop >= 0.965)).sum())

# reference: FT-NMPC v2 from the main Monte Carlo
ft = M[M.ctrl == "FT-NMPC-v2"].set_index("case")
chk = L.set_index("case")
out["reproduces_main_run"] = bool(np.allclose(chk.XE_stop.sort_index(), ft.XE_stop.loc[chk.index].sort_index(), atol=1e-6))


def compare(name, base_on, base_idx):
    g = R[R.ctrl == name].set_index("case"); idx = g.index.intersection(base_idx)
    a = g.on_spec.loc[idx].astype(bool); b = base_on.loc[idx].astype(bool)
    better, worse = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
    d = dict(n=int(len(idx)), on_spec=float(a.mean()), ref_on_spec=float(b.mean()), better=better, worse=worse, p=float(p),
             t_stop=float(g.t_stop.loc[idx].mean()), XE=float(100 * g.XE_stop.loc[idx].mean()),
             cat_kg=float(g.cat_topup_kg.loc[idx].mean()), meoh_kg=float(g.meoh_makeup_kg.loc[idx].mean()))
    return d, g.loc[idx]


ft_on = ft.on_spec
for name in ["Oracle-v2", "FT-v2-batch"]:
    if (R.ctrl == name).any():
        d, g = compare(name, ft_on, ft.index)
        T = table(g.reset_index().assign(oil=[cases[i]["oil"] for i in g.index]), BASE)
        Tf = table(ft.loc[g.index].reset_index(), BASE)
        d["margin"] = float(T.margin.mean()); d["prod"] = float(T["prod"].mean())
        d["ref_margin"] = float(Tf.margin.mean()); d["ref_prod"] = float(Tf["prod"].mean())
        d["by_class"] = g.groupby("cls").on_spec.mean().round(3).to_dict()
        out[name] = d
# remaining off-spec under the oracle vs FT
if "Oracle-v2" in out:
    o = R[R.ctrl == "Oracle-v2"].set_index("case").on_spec.astype(bool)
    f = ft_on.loc[o.index].astype(bool)
    out["Oracle-v2"]["both_off"] = int((~o & ~f).sum())

sub = [ci * 30 + j for ci in range(8) for j in range(6)]
out["thresholds"] = {"base": dict(on_spec=float(ft_on.loc[sub].mean()), t_stop=float(ft.t_stop.loc[sub].mean()))}
for name in sorted(n for n in R.ctrl.unique() if n.startswith("thr:")):
    d, _ = compare(name, ft_on, pd.Index(sub)); out["thresholds"][name] = d
json.dump(out, open(os.path.join(HERE, "revision_summary.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
