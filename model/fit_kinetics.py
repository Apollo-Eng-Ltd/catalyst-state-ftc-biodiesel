"""Fit the plant model to the 2013 yields.

Observation model: measured % FAME yield (gravimetric) = 100 X_E(t_end) + e,
e ~ N(0, sigma^2), sigma = 1.0 point (between-campaign repeatability).
Kinetics k1-k6 and Ea fixed at Noureddini and Zhu (1997). Fitted parameters are
listed in FIT. Usage: python3 fit_kinetics.py [case]
"""
import sys, json, time, datetime, numpy as np, pandas as pd
from scipy.optimize import least_squares
from plant import Params, initial_state, simulate, X_E, I_V, IX, preneutralise

np.random.seed(0)
CASE = sys.argv[1] if len(sys.argv) > 1 else "base"
SIGMA = 1.0
MW_OIL = dict(CSFO=848.3, CWVO=848.3, SBFO=874.0, SWVO=874.0, BFVO=861.2, BWVO=861.2, JCO=870.0)

pts = pd.read_csv("../data/boakye2013_yield_points.csv")
pts = pts[pts.use_in_fit].reset_index(drop=True)
if CASE in ("no1wt", "veg", "jco", "wash_veg", "wash_jco", "wash_all", "final_veg", "final_jco", "hyd_veg", "hyd_jco", "lit_veg", "lit_jco", "lit_jco_kH"):
    pts = pts[pts.cat_wt != 1].reset_index(drop=True)
if CASE in ("veg", "wash_veg", "final_veg", "hyd_veg", "lit_veg"):    # six soybean/cottonseed oils, fresh and waste
    pts = pts[pts.oil != "JCO"].reset_index(drop=True)
if CASE in ("jco", "wash_jco", "final_jco", "hyd_jco", "lit_jco", "lit_jco_kH"):
    pts = pts[pts.oil == "JCO"].reset_index(drop=True)

# one simulation per distinct batch recipe; time points read off the trajectory
pts["recipe"] = pts.apply(lambda r: (r.oil, r.cat_wt, r.MeOH_ratio, r.T_C), axis=1)
recipes = {rc: g for rc, g in pts.groupby("recipe")}

# name: (initial, lower, upper, log-scaled)
FIT = {
    "a":      (0.3,  1e-3, 10.0, True),
    "GL_sat": (0.05, 1e-5, 1.0,  True),
    "c_th":   (0.04, 0.0,  0.2,  False),
    "k_SE":   (1e-3, 1e-6, 1.0,  True),
    "Ea_SE":  (6e4,  0.0,  2e5,  False),
}
if CASE.startswith("hyd"):    # hydroxide pathway: a, c_th, k_SE (on OH), k_H
    FIT.pop("GL_sat"); FIT.pop("Ea_SE")
    FIT["k_SE"] = (0.1, 1e-4, 100.0, True)
    FIT["k_H"] = (1e-3, 1e-6, 1.0, True)
if CASE.startswith("lit"):    # hydroxide pathway, k_SE and Ea_SE fixed from Eze, Harvey, Phan (2015) Fuel 140:724
    FIT.pop("GL_sat"); FIT.pop("Ea_SE"); FIT.pop("k_SE")
    FIT["k_H"] = (1e-2, 1e-6, 1.0, True)
FIXED = dict(k_SE=0.032, Ea_SE=62.0e3) if CASE.startswith("lit") else {}
if CASE == "lit_jco_kH":      # Jatropha shares the K2CO3/MeOH hydroxide chemistry: k_H from the vegetable-oil fit
    FIT.pop("k_H")
    FIXED["k_H"] = json.load(open("fit_kinetics_lit_veg.run.json"))["theta"]["k_H"]
if CASE.startswith("final"):  # identifiable set only: a, c_th, k_SE
    FIT.pop("GL_sat"); FIT.pop("Ea_SE")
if CASE.startswith("wash"):   # soap-driven wash loss in the observation model; GL_sat, Ea_SE fixed
    FIT.pop("GL_sat"); FIT.pop("Ea_SE")
    FIT["beta"] = (1.0, 0.0, 20.0, False)
if CASE == "feed_a":   # separate activity for Jatropha (stored oil: waxes, phospholipids)
    FIT["a_JCO"] = (1.0, 0.1, 2.0, False)
names = list(FIT)

def to_x(theta): return np.array([np.log(theta[n]) if FIT[n][3] else theta[n] for n in names])
def to_theta(x): return {**FIXED, **{n: (np.exp(v) if FIT[n][3] else v) for n, v in zip(names, x)}}

def predict(theta, return_frame=False):
    pred = np.full(len(pts), np.nan)
    for (oil, cat, ratio, T), g in recipes.items():
        a_oil = theta.get("a_JCO", 1.0) if oil == "JCO" else 1.0
        p = Params(a=theta["a"] * a_oil, GL_sat=theta.get("GL_sat", 1e-3), c_th=theta["c_th"],
                   k_SE=theta["k_SE"], Ea_SE=theta.get("Ea_SE", 0.0), k_H=theta.get("k_H", 0.0),
                   k_N=0.0, k_S=0.0)
        AV = g.oil_AV.iloc[0]
        x0 = initial_state(m_oil_g=100, MW_oil=MW_OIL[oil], ratio=ratio, cat_g=cat, AV=AV,
                           water_wt=0.0, T_C=T)
        x0 = preneutralise(x0)
        s = simulate(x0, p, g.t_min.max(), method="LSODA", rtol=1e-6, atol=1e-10)
        for i, r in g.iterrows():
            xt = s.sol(r.t_min)
            # wash loss: fraction of ester lost with soap emulsions = beta * soap / (3 TG0)
            loss = theta.get("beta", 0.0) * xt[IX["Soap"]] / (3 * x0[IX["TG"]])
            pred[i] = 100 * X_E(xt, x0) * (1 - min(loss, 1.0)) if s.success else 0.0
    return pred

def resid(x):
    return (predict(to_theta(x)) - pts["mean"].values) / SIGMA

t0 = time.time()
x_init = to_x({n: FIT[n][0] for n in names})
lb = to_x({n: FIT[n][1] if not FIT[n][3] else FIT[n][1] for n in names})
ub = to_x({n: FIT[n][2] for n in names})
best = None
for trial in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3):  # multistart, seeded
    xs = x_init if trial == 0 else lb + (ub - lb) * np.random.rand(len(names))
    r = least_squares(resid, xs, bounds=(lb, ub), x_scale="jac", diff_step=1e-3, max_nfev=120)
    print(f"start {trial}: cost {r.cost:.2f}  {to_theta(r.x)}", flush=True)
    if best is None or r.cost < best.cost:
        best = r
theta = to_theta(best.x)
pts["pred"] = predict(theta)
res = pts.pred - pts["mean"]
rmse = float(np.sqrt(np.mean(res ** 2)))
r2 = float(1 - np.sum(res ** 2) / np.sum((pts["mean"] - pts["mean"].mean()) ** 2))

# approximate 95 % CIs from the Jacobian (in fit coordinates, mapped back)
J = best.jac
dof = len(pts) - len(names)
s2 = 2 * best.cost / dof
try:
    cov = np.linalg.inv(J.T @ J) * s2
    se = np.sqrt(np.diag(cov))
    ci = {n: ([float(np.exp(best.x[i] - 1.96 * se[i])), float(np.exp(best.x[i] + 1.96 * se[i]))] if FIT[n][3]
              else [float(best.x[i] - 1.96 * se[i]), float(best.x[i] + 1.96 * se[i])]) for i, n in enumerate(names)}
    corr = (cov / np.outer(se, se)).round(2).tolist()
except np.linalg.LinAlgError:
    ci, corr = None, None

by_series = pts.assign(res=res).groupby("series").res.apply(lambda v: float(np.sqrt(np.mean(v ** 2)))).round(2).to_dict()
by_oil = pts.assign(res=res).groupby("oil").res.apply(lambda v: float(np.sqrt(np.mean(v ** 2)))).round(2).to_dict()
summary = dict(case=CASE, run_utc=datetime.datetime.now(datetime.UTC).isoformat(), n=len(pts), sigma=SIGMA,
               theta=theta, ci95=ci, corr=corr, names=names, RMSE=rmse, R2=r2,
               RMSE_excl_1wt=float(np.sqrt(np.mean(res[pts.cat_wt != 1] ** 2))),
               RMSE_by_series=by_series, RMSE_by_oil=by_oil, seconds=round(time.time() - t0, 1),
               chi2_red=float(2 * best.cost / dof))
print(json.dumps(summary, indent=1, default=float))
json.dump(summary, open(f"fit_kinetics_{CASE}.run.json", "w"), indent=1, default=float)
pts.drop(columns="recipe").to_csv(f"fit_kinetics_{CASE}_pred.csv", index=False)
