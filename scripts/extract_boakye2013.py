"""Yield data from the 2013 experimental study (Boakye, KNUST).

Replicate % yields are transcribed from Appendix B (Tables B.1.1-B.3), which list
actual and theoretical FAME masses for each of three runs. Means and 95 % CIs are
recomputed here and compared with the summary Tables 7-10.

% FAME yield (Eq. 4 of the study) = washed product mass / theoretical FAME mass x 100.
It is a gravimetric yield, not a GC ester content.
"""
import json, hashlib, datetime
import numpy as np, pandas as pd
from scipy import stats

OUT = "../data/"

# Oil characterisation, Table 6 / Table A.1.1 (acid value mg KOH/g, % FFA as oleic)
OILS = {
    "CSFO": dict(name="Cottonseed fresh oil",          AV=0.452, FFA=0.227, MW=848.3),
    "SBFO": dict(name="Soya bean fresh oil",           AV=0.396, FFA=0.199, MW=874.0),
    "BFVO": dict(name="Blend of fresh veg. oils",      AV=0.441, FFA=0.222, MW=np.nan),
    "CWVO": dict(name="Cottonseed waste veg. oil",     AV=0.920, FFA=0.462, MW=848.3),
    "SWVO": dict(name="Soya bean waste veg. oil",      AV=1.088, FFA=0.547, MW=874.0),
    "BWVO": dict(name="Blend of waste veg. oils",      AV=1.017, FFA=0.511, MW=np.nan),
    # Jatropha: raw 11.298 mg KOH/g (5.679 % FFA); acid-esterified to 1.31 % FFA before
    # transesterification. AV of the charged oil scaled by the study's own AV/FFA ratio.
    "JCO":  dict(name="Jatropha curcas oil (esterified)", AV=round(1.31 * 11.298 / 5.679, 3),
                 FFA=1.31, MW=870.0),
}

# (series, source table, oil, cat wt%, oil:MeOH, T degC, t min, [rep % yields], reported mean, reported CI)
R = []
def add(series, tab, oil, cat, ratio, T, t, reps, mean, ci):
    R.append(dict(series=series, table=tab, oil=oil, cat_wt=cat, MeOH_ratio=ratio,
                  T_C=T, t_min=t, reps=reps, rep_mean=mean, rep_ci=ci))

# Table 7 / B.1: catalyst loading, 1:6, 60 C, 120 min, 600 rpm
cat = {
 "BFVO": [([14.49,13.93,14.54],14.32,.85),([90.11,90.09,90.07],90.09,.05),([93.21,93.28,93.32],93.27,.14),
          ([96.99,96.97,97.06],97.01,.12),([94.48,94.46,94.47],94.47,.02),([93.56,93.44,93.41],93.47,.20),
          ([93.00,93.03,93.04],93.03,.05)],
 "BWVO": [([11.11,11.08,11.09],11.09,.04),([89.45,89.33,88.58],82.12,1.17),([92.41,91.57,92.31],92.10,1.14),
          ([95.31,95.44,95.29],95.35,.20),([93.42,93.47,93.35],93.41,.15),([93.17,93.12,93.10],93.13,.09),
          ([92.70,92.56,92.11],92.46,.78)],
 "JCO":  [([10.45,10.46,10.48],10.47,.04),([80.74,80.64,80.63],80.67,.15),([86.34,86.32,86.31],86.33,.04),
          ([90.29,90.30,90.31],90.30,.02),([89.21,89.21,89.24],89.22,.04),([86.95,86.94,86.92],86.94,.04),
          ([85.18,85.20,85.19],85.19,.02)],
}
for oil, rows in cat.items():
    for w, (reps, m, ci) in zip(range(1, 8), rows):
        add("catalyst", "7/B.1", oil, w, 6, 60, 120, reps, m, ci)

# Table 8 / B.2: oil:MeOH ratio, 4 wt%, 60 C, 120 min
ratio = {
 "CWVO": [([82.32,82.14,82.49],82.32,.43),([93.30,92.81,93.18],93.09,.63),([91.91,92.09,92.40],92.14,.61)],
 "CSFO": [([87.96,88.29,88.00],88.08,.45),([97.79,97.76,97.70],97.75,.11),([95.26,95.04,95.59],95.29,.68)],
 "SWVO": [([88.11,87.95,88.03],88.03,.21),([95.88,95.97,95.31],95.72,.89),([92.93,92.59,92.99],92.84,.53)],
 "SBFO": [([90.44,90.31,90.48],90.41,.22),([96.49,96.35,96.41],96.41,.17),([94.35,94.10,94.26],94.23,.31)],
 "JCO":  [([68.93,68.45,68.28],68.55,.85),([86.50,87.03,86.63],86.72,.68),([94.45,94.67,94.55],94.56,.29)],
}
for oil, rows in ratio.items():
    for r, (reps, m, ci) in zip([4, 6, 8], rows):
        add("ratio", "8/B.2", oil, 4, r, 60, 120, reps, m, ci)

# Table 9 / B.3: time, 4 wt%, 60 C; 1:6 for waste oils, 1:8 for JCO
time_ = {
 "CWVO": (6, [([98.09,98.56,98.17],98.27,.62),([97.53,97.74,97.76],97.68,.31),([93.29,92.80,93.17],93.09,.63)]),
 "SWVO": (6, [([97.74,97.49,97.30],97.51,.55),([98.50,98.44,98.52],98.48,.10),([95.88,95.97,95.31],95.72,.89)]),
 "JCO":  (8, [([86.32,86.09,86.14],86.19,.30),([88.23,88.12,88.20],88.19,.14),([94.56,94.56,94.60],94.57,.07)]),
}
for oil, (r, rows) in time_.items():
    for t, (reps, m, ci) in zip([40, 80, 120], rows):
        add("time", "9/B.3", oil, 4, r, 60, t, reps, m, ci)

# Table 10 / B.3 (temperature): at each oil's optimum time
temp = {
 "CWVO": (6, 40,  [([99.03,99.15,99.07],99.09,.15),([99.48,99.46,99.43],99.46,.06),([93.29,92.80,93.17],93.09,.63)]),
 "SWVO": (6, 80,  [([98.26,98.20,97.98],98.15,.36),([99.50,99.52,99.51],99.51,.02),([95.88,95.97,95.31],95.72,.89)]),
 "JCO":  (8, 120, [([87.32,87.51,87.70],87.51,.47),([94.55,94.61,94.56],94.57,.09),([89.85,90.15,90.01],90.00,.37)]),
}
for oil, (r, t, rows) in temp.items():
    for T, (reps, m, ci) in zip([55, 60, 65], rows):
        add("temperature", "10/B.3", oil, 4, r, T, t, reps, m, ci)

df = pd.DataFrame(R)
tcrit = stats.t.ppf(0.975, 2)
df["mean"] = df.reps.apply(np.mean).round(2)
df["sd"] = df.reps.apply(lambda x: np.std(x, ddof=1)).round(3)
df["ci95"] = (tcrit * df.sd / np.sqrt(3)).round(2)
df["mean_diff_vs_table"] = (df["mean"] - df.rep_mean).round(2)
for k in ["AV", "FFA"]:
    df[f"oil_{k}"] = df.oil.map(lambda o: OILS[o][k])
# replicate %s are rounded from the same masses, so match to 0.05 points
grp = list(range(len(df)))
for i in range(len(df)):
    for j in range(i):
        if np.all(np.abs(np.array(df.reps[i]) - np.array(df.reps[j])) <= 0.015):
            grp[i] = grp[j]; break
df["rep_key"] = grp

# ---- flags -------------------------------------------------------------------
df["flag"] = ""
df["use_in_fit"] = True
def flag(mask, text, use=None):
    df.loc[mask, "flag"] = (df.loc[mask, "flag"] + "; " + text).str.strip("; ")
    if use is not None:
        df.loc[mask, "use_in_fit"] = use

# 1. Table 7 BWVO 2 wt%: summary value disagrees with its own replicates
m = (df.series == "catalyst") & (df.oil == "BWVO") & (df.cat_wt == 2)
flag(m, "Table 7 prints 82.12; Appendix B.1.2 replicates give 89.12 (transcription error in Table 7; replicate mean used)")

# 2. Identical replicate triplets appearing under different nominal conditions
dup = df.groupby("rep_key").filter(lambda g: len(g) > 1)
for key, g in dup.groupby("rep_key"):
    first = g.index[0]
    c0 = (df.at[first, "T_C"], df.at[first, "t_min"], df.at[first, "MeOH_ratio"])
    for i in g.index[1:]:
        ci_ = (df.at[i, "T_C"], df.at[i, "t_min"], df.at[i, "MeOH_ratio"])
        if ci_ == c0:
            flag(df.index == i, f"same runs as row {first} (identical replicates, identical conditions); duplicate dropped", use=False)
        else:
            flag(df.index == i, f"replicates identical to row {first} (60 C/120 min/1:6) but labelled {ci_[0]} C/{ci_[1]} min; not independent data, excluded", use=False)

# 3. Same nominal conditions, independent triplicates, different means (between-campaign scatter)
pairs = [("CWVO", 6, 60, 40), ("SWVO", 6, 60, 80), ("JCO", 8, 60, 120), ("JCO", 6, 60, 120)]
repro = []
for oil, r, T, t in pairs:
    g = df[(df.oil == oil) & (df.MeOH_ratio == r) & (df.T_C == T) & (df.t_min == t) & (df.cat_wt == 4)]
    g = g.drop_duplicates("rep_key")
    if len(g) > 1:
        repro.append(dict(oil=oil, ratio=r, T=T, t=t, tables=list(g.table), means=list(g["mean"]),
                          spread=round(g["mean"].max() - g["mean"].min(), 2)))
        flag(g.index, f"repeat campaign at same nominal conditions; spread {repro[-1]['spread']} points")

df_out = df.drop(columns=["rep_key"]).copy()
df_out["reps"] = df_out.reps.apply(lambda x: " | ".join(f"{v:.2f}" for v in x))
df_out = df_out.rename(columns={"rep_mean": "table_mean", "rep_ci": "table_ci95", "reps": "replicates_pct"})
cols = ["series", "table", "oil", "oil_AV", "oil_FFA", "cat_wt", "MeOH_ratio", "T_C", "t_min",
        "replicates_pct", "mean", "sd", "ci95", "table_mean", "table_ci95", "mean_diff_vs_table",
        "use_in_fit", "flag"]
df_out[cols].to_csv(OUT + "boakye2013_yield_points.csv", index=True, index_label="row")

long = df.explode("reps")[["series", "table", "oil", "oil_AV", "cat_wt", "MeOH_ratio", "T_C", "t_min", "reps", "use_in_fit"]]
long = long.rename(columns={"reps": "yield_pct"})
long.to_csv(OUT + "boakye2013_yield_replicates.csv", index_label="point_row")

# ---- checks ----------------------------------------------------------------
ci_mismatch = df[(df.ci95 - df.rep_ci).abs() > 0.1][["series", "oil", "cat_wt", "MeOH_ratio", "T_C", "t_min", "ci95", "rep_ci"]]
mean_mismatch = df[df.mean_diff_vs_table.abs() > 0.02][["series", "oil", "cat_wt", "MeOH_ratio", "T_C", "t_min", "mean", "rep_mean"]]

# FFA neutralisation capacity at 1 wt% K2CO3 (100 g oil; K2CO3 -> KHCO3, 1 eq/mol)
neut = {}
for oil in ["BFVO", "BWVO", "JCO"]:
    ffa_mmol = OILS[oil]["AV"] * 100 / 56.106
    neut[oil] = dict(FFA_mmol=round(ffa_mmol, 2), K2CO3_1g_mmol=round(1000 / 138.205, 2),
                     fraction_consumed=round(ffa_mmol / (1000 / 138.205), 3))

print("Points:", len(df), " usable:", int(df.use_in_fit.sum()))
print("\nMean mismatches vs summary tables:\n", mean_mismatch.to_string())
print("\nCI mismatches (>0.1) vs summary tables:\n", ci_mismatch.to_string())
print("\nReproducibility:\n", json.dumps(repro, indent=1))
print("\nNeutralisation at 1 wt%:\n", json.dumps(neut, indent=1))
print("\nFlags:\n", df_out.loc[df_out.flag != "", ["series", "oil", "cat_wt", "MeOH_ratio", "T_C", "t_min", "use_in_fit", "flag"]].to_string())

log = dict(step="extract", run_utc=datetime.datetime.now(datetime.UTC).isoformat(),
           source="Boakye P. (2013) KNUST, Tables 6-10 and Appendix B",
           n_points=len(df), n_usable=int(df.use_in_fit.sum()), tcrit=tcrit,
           oils=OILS, reproducibility=repro, neutralisation_1wt=neut,
           csv_sha1={f: hashlib.sha1(open(OUT + f, "rb").read()).hexdigest()
                     for f in ["boakye2013_yield_points.csv", "boakye2013_yield_replicates.csv"]})
json.dump(log, open(OUT + "boakye2013_yield_points.run.json", "w"), indent=1, default=float)
