"""Evaluate estimators on the held-out splits.

Metrics (t >= 2 min, physical units), per split and per fault class:
  X_E           RMSE in percentage points of conversion
  cat_act       RMSE of active methoxide concentration, mol/L, and as % of the batch's initial base
  a_rel, eta    RMSE of the activity and agitation multipliers
  phys_resid    mean squared species-balance residual of the estimated trajectory (physical consistency)
Estimators: est_pinn.pt, est_data.pt, and EKF (ekf_<split>.npz).
Usage: python3 evaluate.py
"""
import os, sys, json, datetime
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch_plant as tp                                  # noqa: E402
from train_estimator import load, Net, STRIDE, DT, MEAS   # noqa: E402

SPLITS = ["test_interp", "test_extrap", "test_jco"]
TAGS = ["data", "pinn", "pinn2", "data10", "pinn10", "data_adm", "data_ad", "pinn2_ad", "ft1", "ft01", "pinn3", "pinn3_10", "pinn3_ad"]
I0 = int(2.0 / DT)


def net_predict(tag, d):
    ck = torch.load(os.path.join(HERE, f"est_{tag}.pt"), weights_only=False)
    net = Net(ck["n_in"], ck["n_out"]).double(); net.load_state_dict(ck["state"]); net.eval()
    xin = np.concatenate([d["Xm"], np.repeat(d["st"][:, None, :], d["Xm"].shape[1], 1)], -1)
    with torch.no_grad():
        y = net(torch.tensor((xin - ck["mu_x"]) / ck["sd_x"])).numpy() * ck["sd_y"] + ck["mu_y"]
    nS = len(tp.SP)
    z, zV = y[..., :nS], y[..., nS]
    XE = z[..., tp.IX["E"]] / 3.0
    cat = z[..., tp.IX["Cat"]] * d["nTG0"][:, None] / (zV * d["V0"][:, None])
    return dict(X_E=XE, cat_act=cat, a_rel=y[..., nS + 5], eta=y[..., nS + 2], z=z, zV=zV, That=y[..., nS + 1])


def truth(d):
    XE = d["z"][..., tp.IX["E"]] / 3.0
    cat = d["z"][..., tp.IX["Cat"]] * d["nTG0"][:, None] / (d["zV"] * d["V0"][:, None])
    return dict(X_E=XE, cat_act=cat, a_rel=d["pars"][..., 3], eta=d["pars"][..., 0])


def phys_resid(est, d):
    z, zV, T = (torch.tensor(est[k]) for k in ["z", "zV", "That"])
    n0, V0 = torch.tensor(d["nTG0"])[:, None], torch.tensor(d["V0"])[:, None]
    ones = torch.ones_like(zV[:, 1:-1])
    dz, _, _ = tp.dz_dt(z[:, 1:-1], zV[:, 1:-1], T[:, 1:-1], ones, ones, 0 * ones, 0 * ones, n0, V0)
    fd = (z[:, 2:] - z[:, :-2]) / (2 * DT)
    return float((((fd - dz) ** 2)[:, I0:]).mean())


def rmse(a, b, mask=None):
    e = (a - b)[:, I0:]
    return float(np.sqrt(np.nanmean(e ** 2)))


def main():
    rep = dict(run_utc=datetime.datetime.now(datetime.UTC).isoformat(), splits={})
    for sp in SPLITS:
        d = load(sp); tr = truth(d); meta = d["meta"]
        base0 = d["z"][:, 0, tp.IX["Cat"]] * d["nTG0"] / d["V0"]
        res = {}
        for tag in [t for t in TAGS if os.path.exists(os.path.join(HERE, f"est_{t}.pt"))]:
            e = net_predict(tag, d)
            row = dict(X_E_pp=100 * rmse(e["X_E"], tr["X_E"]), cat_molL=rmse(e["cat_act"], tr["cat_act"]),
                       cat_pct_of_base=100 * float(np.sqrt(np.nanmean(((e["cat_act"] - tr["cat_act"])[:, I0:] / base0[:, None]) ** 2))),
                       a_rel=rmse(e["a_rel"], tr["a_rel"]), eta=rmse(e["eta"], tr["eta"]),
                       phys_resid=phys_resid(e, d))
            row["by_class_X_E_pp"] = {c: round(100 * rmse(e["X_E"][meta.cls == c], tr["X_E"][meta.cls == c]), 2) for c in meta.cls.unique()}
            row["by_class_cat_pct"] = {c: round(100 * float(np.sqrt(np.nanmean(((e["cat_act"] - tr["cat_act"])[meta.cls == c][:, I0:] / base0[meta.cls == c][:, None]) ** 2))), 2) for c in meta.cls.unique()}
            res[tag] = row
        fe = os.path.join(HERE, f"ekf_{sp}.npz")
        if os.path.exists(fe):
            k = np.load(fe); sel = k["sel"]; est = k["est"]
            sub = lambda a: a[sel]
            res["ekf"] = dict(n=int(len(sel)), diverged=int(np.isnan(est[:, -1, 0]).sum()), X_E_pp=100 * rmse(est[..., 0], sub(tr["X_E"])),
                              cat_molL=rmse(est[..., 1], sub(tr["cat_act"])),
                              cat_pct_of_base=100 * float(np.sqrt(np.nanmean(((est[..., 1] - sub(tr["cat_act"]))[:, I0:] / base0[sel][:, None]) ** 2))),
                              a_rel=rmse(est[..., 2], sub(tr["a_rel"])), eta=rmse(est[..., 3], sub(tr["eta"])),
                              by_class_X_E_pp={c: round(100 * rmse(est[..., 0][meta.cls.values[sel] == c], sub(tr["X_E"])[meta.cls.values[sel] == c]), 2)
                                               for c in meta.cls.unique()})
        rep["splits"][sp] = res
        print(sp, json.dumps({t: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("by_")} for t, r in res.items()}))
    json.dump(rep, open(os.path.join(HERE, "evaluate.run.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
