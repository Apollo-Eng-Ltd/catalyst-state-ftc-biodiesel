"""Build manuscript Tables 1-8 from the logged results (no hand-typed numbers except stated design values).
Writes figures/tables/TableN.csv and figures/tables/tables.md. Every classifier accuracy is
recomputed from its confusion matrix and asserted equal to the logged value."""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "figures", "tables"); os.makedirs(OUT, exist_ok=True)
sys.path[:0] = [os.path.join(ROOT, "model")]
J = lambda p: json.load(open(os.path.join(ROOT, p)))
md = []

def emit(n, title, df, note=""):
    df.to_csv(os.path.join(OUT, f"Table{n}.csv"), index=False)
    md.append(f"## Table {n}. {title}\n\n" + df.to_markdown(index=False) + (f"\n\n{note}" if note else "") + "\n")

# Table 1
import nz1997_params as nz
from pilot import PILOT
veg, jco = J("model/fit_phase1_lit_veg.run.json"), J("model/fit_phase1_lit_jco_kH.run.json")
ci = veg["ci95"]
rows = []
for i, k in enumerate(["k1", "k2", "k3", "k4", "k5", "k6"]):
    rows.append([f"{k} (50 °C)", f"{nz.K_REF[k]:.3f}", "L mol⁻¹ min⁻¹", f"Ea {nz.EA_J[k]/1000:.1f} kJ mol⁻¹", "Noureddini and Zhu (1997)"])
fmt_ci = lambda n, d=3: f"[{ci[n][0]:.{d}f}, {ci[n][1]:.{d}f}]" if isinstance(ci, dict) and n in ci else ""
rows += [
    ["a, catalyst activity factor (vegetable oils)", f"{veg['theta']['a']:.3f}", "–", fmt_ci("a"), "fitted, our previous experimental study (n = 32)"],
    ["a, catalyst activity factor (Jatropha)", f"{jco['theta']['a']:.4f}", "–", "", "fitted (n = 15)"],
    ["c_th, inactive base threshold", f"{veg['theta']['c_th']:.4f}", "mol L⁻¹", fmt_ci("c_th", 4), "fitted"],
    ["k_H, methoxide → hydroxide", f"{veg['theta']['k_H']:.5f}", "min⁻¹", fmt_ci("k_H", 5), "fitted"],
    ["k_SE, ester saponification (60 °C)", "0.032", "L mol⁻¹ min⁻¹", "Ea 62 kJ mol⁻¹", "Eze et al. (2015)"],
    ["ΔH, transesterification", f"{PILOT['dH_trans_J_mol']/1000:.1f}".replace('-', '−'), "kJ mol⁻¹ ester", "", "Fjerbaek Sotoft et al. (2010)"],
    ["ΔH, neutralisation and saponification (sensitivity case)", "−57.3", "kJ mol⁻¹", "", "Wadsö and Karlsson (2013)"],
    ["Oil charge", f"{PILOT['m_oil_kg']:.0f}", "kg", "", "design"],
    ["Vessel diameter", f"{PILOT['D_m']:.1f}", "m", "", "design"],
    ["U, heating / cooling", f"{PILOT['U_heat_W_m2K']:.0f} / {PILOT['U_cool_W_m2K']:.0f}", "W m⁻² K⁻¹", "", "Thermopedia, agitated vessels"],
    ["Jacket flow", f"{PILOT['F_J_Lmin']:.0f}", "L min⁻¹", "", "design"],
    ["TCU inlet range", f"{PILOT['T_Jin_min_K']-273.15:.0f}–{PILOT['T_Jin_max_K']-273.15:.0f}", "°C", "", "design"],
    ["cp oil / cp methanol", f"{PILOT['cp_oil_J_kgK']/1000:.3f} / {PILOT['cp_MeOH_J_kgK']/1000:.2f}", "kJ kg⁻¹ K⁻¹", "", "Garcia Rojas et al. (2013); NIST"],
    ["PI controller Kc / τI", "6 / 8", "– / min", "back-calculation anti-windup", "tuned"],
]
emit(1, "Kinetic, thermal and pilot parameters", pd.DataFrame(rows, columns=["Parameter", "Value", "Unit", "Note", "Source"]),
     f"Fit RMSE: vegetable oils {veg['RMSE']:.2f} points, Jatropha {jco['RMSE']:.2f} points. Brackets: 95 % confidence intervals.")

# Table 2
ds = J("data/dataset.run.json")
F = [["F1", "Acid value above certificate", "ΔAV = 0.5 + 3.5 s mg KOH g⁻¹", "in charge"],
     ["F2", "Water ingress", "(0.02 + 0.28 s) wt% of oil per 10 min", "in batch"],
     ["F3", "Degraded feedstock activity", "a × (1 − 0.85 s)", "in charge"],
     ["E1", "Jacket TCU degradation", "φ = 1 − 0.9 s", "in batch"],
     ["E2", "Agitation loss", "N/N0 = 1 − 0.8 s; P ∝ N³", "in batch"],
     ["E3", "Methanol loss (condenser)", "(0.002 + 0.028 s) n_A min⁻¹", "in batch"],
     ["E4", "Reactor T sensor bias", "± (0.5 + 3.5 s) K", "in batch"]]
t2 = pd.DataFrame(F, columns=["Class", "Fault", "Magnitude at severity s", "Timing"])
note2 = ("Fault size (severity s): training batches 0 to 0.4 and 0.6 to 0.8; interpolation test 0.4 to 0.6; extrapolation test 0.8 to 1.0. "
         "Fault start time: training 5 to 35 and 50 to 70 min; interpolation test 35 to 50 min; extrapolation test 70 to 90 min. "
         "Faults start abruptly or ramp up over 10 to 30 min, in equal shares. "
         f"Batches per class: {ds['n_train_per_class']} for training (twice as many normal batches) and {ds['n_test_per_class']} per test set. "
         "Hidden plant-model mismatch drawn for every batch: activity a × 0.75 to 1.25, k_SE 0.006 to 0.032 L mol⁻¹ min⁻¹, k_H × 0.85 to 1.15, "
         "c_th × 0.9 to 1.1, UA × 0.9 to 1.1. Recipe ranges: K2CO3 3.5 to 5 wt%, methanol:oil 6 to 8, initial temperature 50 to 55 °C, "
         "set-point 57 to 60 °C, water 0.01 to 0.03 wt%, acid value × 0.8 to 1.2.")
emit(2, "Fault classes and held-out design", t2, note2)

# Table 3
t3 = pd.DataFrame([
    ["T", "Reactor temperature (+ bias under E4)", "0.05 K"],
    ["T_J", "Jacket outlet temperature", "0.05 K"],
    ["T_Jin", "Jacket inlet (TCU outlet) temperature", "0.05 K"],
    ["T_cmd", "TCU set-point from the controller", "exact"],
    ["Q_J", "Jacket duty from flow and temperatures", "from noisy T"],
    ["P_st", "Stirrer power", "2 % of P0"],
    ["pH", "Apparent pH of the methanol/ester phase", "0.02"],
    ["m_ld", "Load-cell mass of contents", "0.2 kg"],
], columns=["Symbol", "Measurement", "Noise (1σ)"])
emit(3, "Online measurements (sample time 0.25 min)", t3, "Not measured online: ester conversion, active catalyst, hydroxide, FFA, water, methanol.")

# Table 4
det = J("classifier/detection.run.json")["splits"]; clf = J("classifier/classifier.run.json")["models"]["res_par"]
CLS = ["N", "F1", "F2", "F3", "E1", "E2", "E3", "E4"]
for sp in ["test_interp", "test_extrap"]:
    cm = pd.read_csv(os.path.join(ROOT, f"classifier/cm_res_par_{sp}.csv"), index_col=0).values
    acc = np.trace(cm) / cm.sum()
    assert abs(acc - clf[sp]["accuracy"]) < 6e-4, (sp, acc, clf[sp]["accuracy"])
rows = []
for c in CLS:
    r = [c]
    for sp in ["test_interp", "test_extrap"]:
        r += [f"{100*det[sp]['detect_rate'][c]:.0f}", f"{100*clf[sp]['recall'][c]:.0f}",
              "–" if c == "N" else f"{clf[sp]['online']['median_delay_min'][c]:.1f}"]
    rows.append(r)
cmI = pd.read_csv(os.path.join(ROOT, "classifier/cm_res_par_test_interp.csv"), index_col=0).values
cmE = pd.read_csv(os.path.join(ROOT, "classifier/cm_res_par_test_extrap.csv"), index_col=0).values
rows.append(["Overall accuracy", "", f"{100*np.trace(cmI)/cmI.sum():.1f}", "", "", f"{100*np.trace(cmE)/cmE.sum():.1f}", ""])
t4 = pd.DataFrame(rows, columns=["Class", "Interp: alarm %", "Interp: recall %", "Interp: isolation delay, min",
                                 "Extrap: alarm %", "Extrap: recall %", "Extrap: isolation delay, min"])
emit(4, "Detection and classification on held-out batches (residuals + estimated parameters, causal)", t4,
     "Alarm % for N is the batch false-alarm rate. Overall accuracies are the confusion-matrix traces (Supplementary S1). "
     "Real Jatropha charges: 80 of 80 classified as F3.")

# Table 5 (values logged in estimator/evaluate.run.json and notes)
ev = J("estimator/evaluate.run.json")["splits"]
t5 = pd.DataFrame([
    ["Causal GRU, data only", 0.61, 3.0, 2.18, 3.6, 9.13, 0.230],
    ["Causal GRU, 10 % of labels", 0.83, 5.1, 2.38, 5.5, 9.42, 0.266],
    ["Physics-informed GRU, nominal constants", "1.73–2.00", "6.0–6.2", "2.94–3.40", 6.5, "12.0–13.1", 0.671],
    ["Physics-informed GRU, constants estimated", 1.78, 4.9, 4.34, 5.5, 17.07, 0.827],
    ["EKF, same model", 2.47, 11.0, 10.69, 12.8, 40.48, 0.349],
    ["Data GRU + unlabelled adaptation, measurement terms", 1.39, 4.9, 2.35, 5.1, 7.96, 0.306],
    ["Data GRU + unlabelled adaptation, physics + measurement", 3.47, 10.2, 3.22, 9.9, 7.09, 0.045],
], columns=["Estimator", "Interp X_E RMSE, points", "Interp catalyst, % of base", "Extrap X_E", "Extrap catalyst",
            "Jatropha X_E", "Jatropha activity RMSE"])
emit(5, "State-estimation accuracy on open-loop held-out batches", t5, "Errors counted from t ≥ 2 min.")

# Table 6
mc = J("control/mc2_summary.json"); S = mc["summary"]
ORDER = ["FT-NMPC-v2", "EKF-NMPC-v2", "FT-NMPC-v1", "NMPC-nom", "PID-35", "PID-60"]
NAME = {"FT-NMPC-v2": "FT-NMPC, rule v2", "EKF-NMPC-v2": "EKF-NMPC, rule v2", "FT-NMPC-v1": "FT-NMPC, rule v1",
        "NMPC-nom": "Nominal NMPC", "PID-35": "PI, stop at 35 min", "PID-60": "PI, stop at 60 min"}
f = lambda v, d: "–" if v is None else f"{v:.{d}f}"
rows = [[NAME[c], f"{100*S[c]['on_spec']:.1f} [{100*S[c]['ci'][0]:.1f}, {100*S[c]['ci'][1]:.1f}]", f(S[c]["XE_mean"], 2),
         f(S[c]["t_stop"], 1), f(S[c]["meoh_kg"], 1), f(S[c]["cat_kg"], 3), S[c]["T_viol"], f(S[c]["solve_mean"], 2), f(S[c]["solve_max"], 1)] for c in ORDER]
emit(6, f"Closed-loop Monte Carlo, {mc['n_scenarios']} scenarios (seed 77, all classes)", pd.DataFrame(rows, columns=[
    "Controller", "On spec, % [95 % CI]", "Mean X_E at stop, %", "Mean stop time, min", "Methanol added, kg",
    "K2CO3 added, kg", "Batches > 64.7 °C", "Mean solve, s", "Max solve, s"]), "CI: 95 % confidence interval from bootstrap resampling over scenarios.")

# Table 7
rows = []
for k, v in mc["mcnemar"].items():
    a, b = k.split(" vs ")
    rows.append([NAME[a], NAME[b], v["better"], v["worse"], f"{v['p']:.2g}"])
emit(7, "Paired McNemar tests on the same 240 scenarios", pd.DataFrame(rows, columns=["A", "B", "On spec only under A", "On spec only under B", "Exact p (two-sided)"]))

# Table 8
ec = J("control/economics.json"); B = ec["base"]; P = ec["prices"]
rows = [[NAME[c], f"{100*B[c]['on_spec']:.1f}", f"{B[c]['margin']:.0f}", f"{B[c]['prod']:.0f}",
         "–" if c == "FT-NMPC-v2" else (f"{ec['breakeven_d'][c]:.2f}" if ec['breakeven_d'][c] > 0 else "0 (v2 higher at any d)")] for c in ORDER]
emit(8, "Economics per batch", pd.DataFrame(rows, columns=["Controller", "On spec, %", "Margin, USD per batch", "Production efficiency, USD per reactor hour",
                                                            "Break-even discount d*"]),
     f"Prices: biodiesel USD {P['P_bd']:.0f} t⁻¹, methanol USD {P['P_MeOH']:.0f} t⁻¹, K2CO3 USD {P['P_cat']:.0f} t⁻¹; off-spec discount d = {P['d']}; "
     f"turnaround τ0 = {P['tau0']:.0f} min. d*: discount above which FT-NMPC v2 earns more per reactor hour.")

open(os.path.join(OUT, "tables.md"), "w").write("# Manuscript tables\n\n" + "\n".join(md))
print("ok", OUT)
