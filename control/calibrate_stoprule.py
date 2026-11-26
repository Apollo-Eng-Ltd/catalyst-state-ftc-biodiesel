"""Calibrate the conversion margin of the stop rule on held-out closed-loop batches.

Data: cl_test_interp (640 closed-loop batches, all eight classes; never used to train est_data_cl and
not part of any Monte Carlo set). Error e = X_E,hat - X_E,true of the causal estimator at every
0.5-min sample with 10 <= t and 0.955 < X_E,hat < 0.99 (the region where the stop decision is taken).
Margins are one-sided 95 % empirical quantiles (split-conformal, finite-sample corrected):
  delta0  samples with no model-free alarm condition (N, F1, F3, E4, and E-class/F2 before onset)
  delta1  samples after onset of the faults the model-free alarms see (F2 water, E3 methanol: load cell;
          E1 TCU: T_cmd - T_Jin; E2 agitation: stirrer power)
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
xh, xe = y[:, :, 3] / 3, d["z"][:, :, 3] / 3
e = xh - xe
m = d["meta"]; t = np.arange(T) * 0.5
cls = m.cls.values.astype(str)[:, None]; tf = m.t_f.values[:, None]
near = (xh > 0.955) & (xh < 0.99) & (t[None] >= 10)
seen = np.isin(cls, ["F2", "E3", "E1", "E2"]) & (t[None] >= tf)


def q(v, a=0.95):
    n = len(v); k = int(np.ceil((n + 1) * a)) - 1
    return float(np.sort(v)[min(k, n - 1)])


out = dict(delta0=q(e[near & ~seen]), delta1=q(e[near & seen]), n0=int((near & ~seen).sum()), n1=int((near & seen).sum()),
           batches=int(len(m)), split="cl_test_interp", estimator="est_data_cl",
           bias_pp=float(100 * e[near].mean()), sd_pp=float(100 * e[near].std()))
json.dump(out, open(os.path.join(HERE, "stoprule_calibration.json"), "w"), indent=1)
print(out)
