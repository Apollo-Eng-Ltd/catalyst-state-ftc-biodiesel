"""Controller-dependent economics per batch, from the fresh Monte Carlo (mc2_results.jsonl).

Only the terms a controller changes are counted (oil, base methanol and base catalyst charge are the same for
every controller and cancel in comparisons):
  margin = P_bd * m_FAME * (1 - d * offspec) - P_MeOH * (1 - eta) * m_MeOH,makeup - P_cat * m_cat,topup
  m_FAME = X_E,stop * 3 * MW_FAME / MW_oil * m_oil          (mass of ester at the stop time)
  productivity = margin / (t_stop + tau_0)                   (USD per reactor hour; tau_0 = charge, settle, drain)
Prices (sourced, Nov 2026) and assumptions:
  P_bd   980 USD/t   B100 FAME, Hamburg, H1 2026, EUR 700-800/m3 = USD 914-1,044/t (Argus Media)
  P_MeOH 540 USD/t   Methanex expected average realized price Oct-Nov 2026, USD 525-550/t (Q3 2026 results)
  P_cat  1,100 USD/t K2CO3, Intratec April 2026: USD 984 (SE Asia) to 1,170 (US) per t
  d      0.20        off-specification batch value discount (assumption; reprocessing or downgrade), 0.1-0.3
  eta    0           methanol make-up recovered (base: none recovered, conservative); 0.95 in sensitivity
  tau_0  90 min      batch turnaround (assumption), 60-180 min in sensitivity
Bootstrap 95 % confidence intervals over scenarios (paired, 2000 resamples).
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
from pilot import MW_OIL, PILOT                                  # noqa: E402
from scenarios import mc_cases                                    # noqa: E402

BASE = dict(P_bd=980.0, P_MeOH=540.0, P_cat=1100.0, d=0.20, eta=0.0, tau0=90.0)


def margin(df, P):
    m_oil = PILOT["m_oil_kg"]
    mw = df.oil.map(MW_OIL)
    m_fame = df.XE_stop * 3 * 292.0 / mw * m_oil / 1000.0                       # t
    v = P["P_bd"] * m_fame * (1 - P["d"] * (~df.on_spec).astype(float))
    c = P["P_MeOH"] * (1 - P["eta"]) * df.meoh_makeup_kg / 1000.0 + P["P_cat"] * df.cat_topup_kg / 1000.0
    return v - c


def table(df, P):
    df = df.copy(); off = (~df.on_spec).astype(float)
    if P.get("rework"):
        # off-spec batch reprocessed: one more pass (30 min + turnaround, 13 kg methanol make-up), then sold at full price
        df["margin"] = margin(df, dict(P, d=0.0)) - off * P["P_MeOH"] * (1 - P["eta"]) * 13.0 / 1000.0
        df["prod"] = df.margin / ((df.t_stop + P["tau0"] + off * (30.0 + P["tau0"])) / 60.0)
    else:
        df["margin"] = margin(df, P); df["prod"] = df.margin / ((df.t_stop + P["tau0"]) / 60.0)
    return df


def boot_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = a - b; idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1); return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


if __name__ == "__main__":
    df = pd.read_json(os.path.join(HERE, "mc2_results.jsonl"), lines=True)
    df = df[df.error.isna()]
    full = df.groupby("case").ctrl.nunique(); keep = full[full == df.ctrl.nunique()].index
    df = df[df.case.isin(keep)]
    out = dict(prices=BASE, n_scenarios=int(len(keep)), base={}, diff_vs_FTv2={}, sens={})
    T = table(df, BASE)
    for k, g in T.groupby("ctrl"):
        out["base"][k] = dict(on_spec=float(g.on_spec.mean()), margin=float(g.margin.mean()), prod=float(g["prod"].mean()),
                              t_stop=float(g.t_stop.mean()), meoh_kg=float(g.meoh_makeup_kg.mean()), cat_kg=float(g.cat_topup_kg.mean()))
    P = T.pivot(index="case", columns="ctrl", values="prod"); M = T.pivot(index="case", columns="ctrl", values="margin")
    for k in P.columns:
        if k != "FT-NMPC-v2":
            out["diff_vs_FTv2"][k] = dict(prod=boot_diff(P["FT-NMPC-v2"].values, P[k].values),
                                         margin=boot_diff(M["FT-NMPC-v2"].values, M[k].values))
    for name, ch in [("d=0.1", dict(d=0.1)), ("d=0.3", dict(d=0.3)), ("eta=0.95", dict(eta=0.95)),
                     ("tau0=60", dict(tau0=60.0)), ("tau0=180", dict(tau0=180.0)),
                     ("P_MeOH=330", dict(P_MeOH=330.0)), ("P_MeOH=1480", dict(P_MeOH=1480.0)),
                     ("P_bd=914", dict(P_bd=914.0)), ("P_bd=1044", dict(P_bd=1044.0)), ("rework", dict(rework=True))]:
        Tk = table(df, dict(BASE, **ch))
        out["sens"][name] = {k: float(g["prod"].mean()) for k, g in Tk.groupby("ctrl")}
    # break-even off-spec discount d* at which FT-NMPC-v2 matches each controller in productivity
    out["breakeven_d"] = {}
    for k in sorted(df.ctrl.unique()):
        if k == "FT-NMPC-v2":
            continue
        f = lambda d: table(df, dict(BASE, d=d)).groupby("ctrl")["prod"].mean()
        g = lambda d: f(d)["FT-NMPC-v2"] - f(d)[k]
        lo, hi = 0.0, 1.0
        if g(lo) >= 0:
            out["breakeven_d"][k] = 0.0; continue
        if g(hi) < 0:
            out["breakeven_d"][k] = None; continue
        for _ in range(40):
            mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
        out["breakeven_d"][k] = round(hi, 3)
    for name in ["rework"]:
        Tk = table(df, dict(BASE, rework=True)); Pk = Tk.pivot(index="case", columns="ctrl", values="prod")
        out["diff_vs_FTv2_rework"] = {k: boot_diff(Pk["FT-NMPC-v2"].values, Pk[k].values) for k in Pk.columns if k != "FT-NMPC-v2"}
    json.dump(out, open(os.path.join(HERE, "economics.json"), "w"), indent=1)
    print(json.dumps(out["base"], indent=1)); print(json.dumps(out["diff_vs_FTv2"], indent=1))
