"""Batch-level split-conformal margins for the stop rule.

The sample-level margins of calibrate_stoprule.py treat every 0.5-min sample as exchangeable, although
samples within one batch are strongly correlated. Here each calibration batch contributes one score, the
largest error X_E,hat - X_E,true over its stop-region samples (before, and after, the onset of a fault the
model-free alarms see). The margins are finite-sample corrected 95 % quantiles over batches.
Output: stoprule_calibration_batch.json
"""
import os, sys, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "estimator"))
sys.argv = [sys.argv[0], "--cl"]
import train_estimator as te

ck = torch.load(os.path.join(HERE, "..", "estimator", "est_data_cl.pt"), weights_only=False)
net = te.Net(ck["n_in"], ck["n_out"]).double(); net.load_state_dict(ck["state"]); net.eval()
d = te.load("test_interp")
T = d["Xm"].shape[1]
xin = np.concatenate([d["Xm"], np.repeat(d["st"][:, None], T, 1)], 2)
with torch.no_grad():
    y = net(torch.tensor((xin - ck["mu_x"]) / ck["sd_x"])).numpy() * ck["sd_y"] + ck["mu_y"]
e = y[:, :, 3] / 3 - d["z"][:, :, 3] / 3
xh = y[:, :, 3] / 3
m = d["meta"]; t = np.arange(T) * 0.5
cls = m.cls.values.astype(str)[:, None]; tf = m.t_f.values[:, None]
near = (xh > 0.955) & (xh < 0.99) & (t[None] >= 10)
seen = np.isin(cls, ["F2", "E3", "E1", "E2"]) & (t[None] >= tf)


def q(v, a=0.95):
    n = len(v); k = int(np.ceil((n + 1) * a)) - 1
    return float(np.sort(v)[min(k, n - 1)])


s0 = [e[i][near[i] & ~seen[i]].max() for i in range(len(m)) if (near[i] & ~seen[i]).any()]
s1 = [e[i][near[i] & seen[i]].max() for i in range(len(m)) if (near[i] & seen[i]).any()]
out = dict(delta0=q(np.array(s0)), delta1=q(np.array(s1)), n0=len(s0), n1=len(s1), split="cl_test_interp",
           estimator="est_data_cl", score="max error per batch in the stop region")
json.dump(out, open(os.path.join(HERE, "stoprule_calibration_batch.json"), "w"), indent=1)
print(out)
