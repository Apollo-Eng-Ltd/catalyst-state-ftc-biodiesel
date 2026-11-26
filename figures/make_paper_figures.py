"""Figures of the paper. One fixed colour per controller across all figures.
Output: figures/paper/FigN_*.png (600 dpi) and .pdf
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "paper"); os.makedirs(OUT, exist_ok=True)
sys.path[:0] = [os.path.join(ROOT, d) for d in ["model", "faults", "control", "estimator", "classifier"]]

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "lines.linewidth": 1.4, "savefig.dpi": 300, "figure.dpi": 150,
                     "axes.grid": True, "grid.color": "#e7e6e2", "grid.linewidth": 0.5})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
CTRL = ["FT-NMPC-v2", "EKF-NMPC-v2", "FT-NMPC-v1", "NMPC-nom", "PID-35", "PID-60"]
LABEL = {"FT-NMPC-v2": "FT-NMPC, rule v2", "EKF-NMPC-v2": "EKF-NMPC, rule v2", "FT-NMPC-v1": "FT-NMPC, rule v1",
         "NMPC-nom": "Nominal NMPC", "PID-35": "PI, 35 min", "PID-60": "PI, 60 min"}
COL = dict(zip(CTRL, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
if os.environ.get("FIGSTYLE", "matlab") == "matlab":   # MATLAB-default look: Arial-metric font, full box, inward ticks, no grid
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.linewidth": 0.5, "axes.edgecolor": "black",
                         "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "text.color": "black",
                         "axes.spines.top": True, "axes.spines.right": True, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True, "xtick.major.size": 3, "ytick.major.size": 3,
                         "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
                         "axes.grid": False, "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
                         "legend.framealpha": 1.0, "lines.linewidth": 1.0, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic"})
    INK = INK2 = "black"
    COL = dict(zip(CTRL, ["#0072BD", "#D95319", "#77AC30", "#EDB120", "#7E2F8E", "#4DBEEE"]))
CLS = ["N", "F1", "F2", "F3", "E1", "E2", "E3", "E4"]
CLSNAME = {"N": "Normal", "F1": "F1 acid value", "F2": "F2 water ingress", "F3": "F3 degraded oil", "E1": "E1 TCU capacity",
           "E2": "E2 agitation", "E3": "E3 methanol loss", "E4": "E4 sensor bias"}
MM = 1 / 25.4


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=600)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def catax(ax, axis):
    """Categorical axis: no minor ticks, no mirrored ticks on the value-label side."""
    if axis == "y":
        ax.tick_params(axis="y", which="both", right=False); ax.tick_params(axis="y", which="minor", left=False)
        ax.tick_params(axis="x", which="both", top=True)
    else:
        ax.tick_params(axis="x", which="minor", bottom=False, top=False); ax.tick_params(axis="x", which="major", top=False)


def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom", color=INK)


# Fig 3 fault trajectories
def fig3():
    from faults import simulate_batch
    base = dict(oil="SWVO", cat_wt=4.0, ratio=6.0, T0_C=53.0, water_wt=0.0002, AV=1.088, Tsp_C=60.0, group="veg",
                s=0.0, t_f=0.0, shape="abrupt", tau_r=20.0, sign=1.0)
    sims = {}
    for c in CLS:
        cc = dict(base, cls=c, s=0.6 if c != "N" else 0.0, t_f=15.0 if c in ("F2", "E1", "E2", "E3", "E4") else 0.0)
        sims[c] = simulate_batch(cc, t_end=60.0, seed=3)
    ref = sims["N"]; t = ref["t"]
    sig = [("X_E", "Conversion (%)", 100, (60, 100)), ("T", "Reactor T (°C)", 1, (50, 66)),
           ("pH", "Apparent pH", 1, (10.6, 12.8)), ("m_ld", "Mass change (kg)", 1, (-45, 25))]
    fig, axs = plt.subplots(4, 7, figsize=(190 * MM, 112 * MM), sharex=True, sharey="row")
    for j, c in enumerate(CLS[1:]):
        o = sims[c]
        for r, (ch, lab, f, yl) in enumerate(sig):
            ax = axs[r, j]
            y0, y = np.asarray(ref[ch]) * f, np.asarray(o[ch]) * f
            if ch == "m_ld":
                y0, y = y0 - y0[0], y - y[0]
            ax.plot(t, y0, color=MUTED, lw=0.9)
            ax.plot(t, y, color=COL["FT-NMPC-v2"], lw=1.1)
            if c in ("F2", "E1", "E2", "E3", "E4"):
                ax.axvline(15, color=INK2, lw=0.6, ls=":")
            ax.set_ylim(*yl); ax.set_xlim(0, 60); ax.set_xticks([0, 30])
            if r == 0:
                ax.set_title(CLSNAME[c].replace(" ", "\n", 1), fontsize=7.2, color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7.2)
            if r == 3:
                ax.set_xlabel("t (min)")
    fig.align_ylabels(axs[:, 0]); fig.subplots_adjust(wspace=0.12, hspace=0.18)
    save(fig, "Fig3_fault_classes")


# Fig 4 detection
def fig4():
    d = json.load(open(f"{ROOT}/classifier/detection.run.json")); h = d["h"]
    R = np.load(f"{ROOT}/data/test_interp_R.npz", allow_pickle=True); D = R["D"]; dt = float(R["dt"])
    m = pd.read_csv(f"{ROOT}/data/test_interp_meta.csv"); t = np.arange(D.shape[1]) * dt
    fig = plt.figure(figsize=(190 * MM, 70 * MM)); gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0]); panel(ax, "(a)")
    show = [("N", "#8a8984"), ("F2", COL["FT-NMPC-v2"]), ("E3", COL["EKF-NMPC-v2"]), ("F3", COL["FT-NMPC-v1"])]
    for c, col in show:
        idx = m[(m.cls == c)]
        i = (idx.s - 0.5).abs().idxmin() if c != "N" else idx.index[0]
        ax.semilogy(t, np.maximum(D[i], 1e-1), color=col, lw=0.9, label=(CLSNAME[c] if c != "N" else "Normal") + f" (onset {m.t_f[i]:.0f} min)" * (c in ("F2", "E3")))
    ax.axhline(h, color=INK, lw=0.8, ls="--"); ax.text(1, h * 1.3, f"threshold h = {h:.1f}", fontsize=7, color=INK)
    ax.set_xlim(0, 75); ax.set_xlabel("t (min)"); ax.set_ylabel("Chi-square statistic D (-)"); ax.legend(loc="upper left", fontsize=6.5, bbox_to_anchor=(0.0, 0.93))
    ax2 = fig.add_subplot(gs[1]); panel(ax2, "(b)")
    cl = [c for c in CLS]; y = np.arange(len(cl))
    ri = [d["splits"]["test_interp"]["detect_rate"][c] for c in cl]; re = [d["splits"]["test_extrap"]["detect_rate"][c] for c in cl]
    ax2.barh(y - 0.19, ri, height=0.36, color=COL["FT-NMPC-v2"], label="held-out, interpolated")
    ax2.barh(y + 0.19, re, height=0.36, color=COL["EKF-NMPC-v2"], label="held-out, extrapolated")
    for k in range(len(cl)):
        ax2.text(ri[k] + 0.01, y[k] - 0.19, f"{100*ri[k]:.0f}", va="center", fontsize=6.5, color=INK2)
        ax2.text(re[k] + 0.01, y[k] + 0.19, f"{100*re[k]:.0f}", va="center", fontsize=6.5, color=INK2)
    ax2.set_yticks(y); ax2.set_yticklabels([CLSNAME[c] if c != "N" else "Normal (false alarm)" for c in cl]); ax2.invert_yaxis()
    ax2.set_xlim(0, 1.15); catax(ax2, "y"); ax2.set_xlabel("Batches with an alarm (fraction)"); ax2.grid(axis="y", visible=False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=6.5)
    save(fig, "Fig4_detection")


# Fig 5 estimation
def fig5():
    import torch
    sys.argv = ["x"]
    import train_estimator as te
    ck = torch.load(f"{ROOT}/estimator/est_data.pt", weights_only=False)
    net = te.Net(ck["n_in"], ck["n_out"]).double(); net.load_state_dict(ck["state"]); net.eval()
    dd = te.load("test_interp"); T = dd["Xm"].shape[1]
    xin = np.concatenate([dd["Xm"], np.repeat(dd["st"][:, None], T, 1)], 2)
    with torch.no_grad():
        y = net(torch.tensor((xin - ck["mu_x"]) / ck["sd_x"])).numpy() * ck["sd_y"] + ck["mu_y"]
    xe_h, xe = 100 * y[:, :, 3] / 3, 100 * dd["z"][:, :, 3] / 3
    fig = plt.figure(figsize=(190 * MM, 68 * MM)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.85], wspace=0.9)
    ax = fig.add_subplot(gs[0]); panel(ax, "(a)")
    k = np.arange(4, T, 10)
    ax.scatter(xe[:, k].ravel(), xe_h[:, k].ravel(), s=1.5, color=COL["FT-NMPC-v2"], alpha=0.25, lw=0, rasterized=True)
    ax.plot([0, 100], [0, 100], color=INK, lw=0.7)
    ax.set_xlim(40, 100); ax.set_ylim(40, 100); ax.set_xlabel("True conversion (%)"); ax.set_ylabel("Estimated conversion (%)")
    rmse = np.sqrt(((xe_h - xe)[:, 4:] ** 2).mean()); ax.text(42, 95, f"RMSE {rmse:.2f} points\n640 held-out batches", fontsize=7, color=INK2, va="top")
    ax2 = fig.add_subplot(gs[1]); panel(ax2, "(b)")
    mm_ = dd["meta"].reset_index(drop=True); drop = (mm_.XE_peak - mm_.XE_end).where(mm_.cls == "F2", -1); i = int(drop.idxmax())
    tt = np.arange(T) * 0.5
    ax2.plot(tt, xe[i], color=INK, lw=1.2, label="true")
    ax2.plot(tt, xe_h[i], color=COL["FT-NMPC-v2"], lw=1.2, ls="--", label="GRU estimate")
    ax2.axvline(dd["meta"].t_f.values[i], color=INK2, lw=0.6, ls=":")
    ax2.set_xlim(0, tt[-1]); ax2.set_ylim(86, 100); ax2.set_yticks([86, 90, 94, 98]); ax2.set_xlabel("t (min)"); ax2.set_ylabel("Ester conversion (%)")
    ax2.set_title("Water ingress batch (F2)", fontsize=7.5); ax2.legend(loc="lower left", bbox_to_anchor=(0.1, 0.02), fontsize=7)
    ax2.text(mm_.t_f[i] + 2, 99.3, f"onset {mm_.t_f[i]:.0f} min", fontsize=6.5, color=INK2)
    ax3 = fig.add_subplot(gs[2]); panel(ax3, "(c)")
    names = ["Data-only GRU", "Adaptive GRU\n(measurement)", "Adaptive GRU\n(physics + meas.)", "EKF"]
    vals = [0.230, 0.306, 0.045, 0.349]
    cols = [MUTED, MUTED, COL["FT-NMPC-v2"], COL["EKF-NMPC-v2"]]
    yb = np.arange(4); ax3.barh(yb, vals, color=cols, height=0.6); catax(ax3, "y")
    for k2 in range(4):
        ax3.text(vals[k2] + 0.008, yb[k2], f"{vals[k2]:.3f}", va="center", fontsize=7, color=INK2)
    ax3.set_yticks(yb); ax3.set_yticklabels(names, fontsize=7); ax3.invert_yaxis(); ax3.set_xlim(0, 0.42)
    ax3.set_xlabel("Activity RMSE on Jatropha charges (-)"); ax3.grid(axis="y", visible=False)
    save(fig, "Fig5_estimation")


# Fig 6 stop-rule calibration
def fig6():
    cal = json.load(open(f"{ROOT}/control/stoprule_calibration.json"))
    z = np.load("/tmp/cl_e.npz"); e, xh = z["e"], z["xe_hat"]; m = pd.read_csv("/tmp/cl_meta.csv")
    t = np.arange(e.shape[1]) * 0.5
    cls = m.cls.values.astype(str)[:, None]; tf = m.t_f.values[:, None]
    near = (xh > 0.955) & (xh < 0.99) & (t[None] >= 10)
    seen = np.isin(cls, ["F2", "E3", "E1", "E2"]) & (t[None] >= tf)
    fig, ax = plt.subplots(figsize=(90 * MM, 62 * MM))
    bins = np.linspace(-3, 3, 61)
    ax.hist(100 * e[near & ~seen], bins=bins, density=True, color=COL["FT-NMPC-v2"], alpha=0.85, label="no alarm")
    ax.hist(100 * e[near & seen], bins=bins, density=True, histtype="step", color=COL["EKF-NMPC-v2"], lw=1.3, label="after an alarm")
    for v, c, lab in [(cal["delta0"], COL["FT-NMPC-v2"], "δ0"), (cal["delta1"], COL["EKF-NMPC-v2"], "δ1")]:
        ax.axvline(100 * v, color=c, lw=1.0, ls="--")
        ax.text(100 * v + 0.05, ax.get_ylim()[1] * 0.92 if lab == "δ0" else ax.get_ylim()[1] * 0.78, (r"$\delta_0$" if lab == "δ0" else r"$\delta_1$") + f" = {100*v:.2f}", color=c, fontsize=7)
    ax.set_xlabel(r"Estimation error, $\hat{X}_E - X_E$ (points)"); ax.set_ylabel("Density (-)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6.5, frameon=False); ax.set_xlim(-3, 3)
    save(fig, "Fig6_stoprule_calibration")


# Fig 7 headline batches
def fig7():
    R = json.load(open(f"{ROOT}/matlab/reference/refs_headline.json"))["refs"]
    R2 = json.load(open(f"{ROOT}/matlab/refs_v2.json"))
    cases = [("normal", "Normal"), ("degraded_oil", "Degraded oil, a = 0.49"),
             ("water_ingress", "Water ingress, 10 min"), ("methanol_loss", "Methanol loss, 10 min")]
    fig, axs = plt.subplots(1, 4, figsize=(190 * MM, 58 * MM), sharey=True)
    series = [("PID35", "PID-35"), ("NMPC", "NMPC-nom"), ("FTNMPC", "FT-NMPC-v1"), ("v2", "FT-NMPC-v2")]
    for j, (c, title) in enumerate(cases):
        ax = axs[j]
        for key, ctrl in series:
            r = R2[c] if key == "v2" else R[c][key]
            ax.plot(r["t"], 100 * np.array(r["X_E"]), color=COL[ctrl], lw=1.3, label=LABEL[ctrl])
            ax.plot(r["t_stop"], 100 * r["XE_stop"], "o", ms=3.5, color=COL[ctrl], mec="white", mew=0.6)
        ax.axhline(96.5, color=INK, lw=0.6, ls=":")
        ax.set_title(f"({'abcd'[j]}) " + title, fontsize=7.5, loc="left"); ax.set_xlim(0, 70); ax.set_ylim(85, 100); ax.set_xlabel("t (min)")
    axs[0].set_ylabel("Ester conversion (%)"); axs[0].text(68, 96.8, "EN 14214", fontsize=6.5, ha="right", color=INK2)
    h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7, frameon=False); fig.subplots_adjust(bottom=0.27)
    save(fig, "Fig7_headline_batches")


# Fig 8 Monte Carlo by class
def fig8():
    s = json.load(open(f"{ROOT}/control/mc2_summary.json"))
    groups = CLS + ["All"]
    fig, ax = plt.subplots(figsize=(190 * MM, 62 * MM))
    w = 0.13; x = np.arange(len(groups))
    for k, c in enumerate(CTRL):
        v = [s["by_class"][g][c] for g in CLS] + [s["summary"][c]["on_spec"]]
        ax.bar(x + (k - 2.5) * w, 100 * np.array(v), width=w * 0.92, color=COL[c], label=LABEL[c]); catax(ax, "x")
    ax.set_xticks(x); SH = {"N": "Normal", "F1": "F1\nacid value", "F2": "F2\nwater", "F3": "F3\ndegraded oil", "E1": "E1\nTCU", "E2": "E2\nagitation", "E3": "E3\nMeOH loss", "E4": "E4\nsensor bias", "All": "All 240\nscenarios"}
    ax.set_xticklabels([SH.get(g, g) for g in groups], fontsize=7)
    ax.set_ylabel("Batches on specification (%)"); ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=7)
    ax.axvline(len(CLS) - 0.5, color=INK2, lw=0.6)
    save(fig, "Fig8_montecarlo_by_class")


# Fig 9 economics
def fig9():
    from economics import table, BASE
    df = pd.read_json(f"{ROOT}/control/mc2_results.jsonl", lines=True); df = df[df.error.isna()]
    T = table(df, BASE)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(90 * MM, 72 * MM))
    for c in CTRL:
        g = T[T.ctrl == c]; mm, pp = g.margin.values, g["prod"].values
        bi = rng.integers(0, len(g), (2000, len(g)))
        cm, cp = np.quantile(mm[bi].mean(1), [0.025, 0.975]), np.quantile(pp[bi].mean(1), [0.025, 0.975])
        ax.errorbar(pp.mean(), mm.mean(), xerr=[[pp.mean() - cp[0]], [cp[1] - pp.mean()]], yerr=[[mm.mean() - cm[0]], [cm[1] - mm.mean()]],
                    fmt="o", ms=5, color=COL[c], ecolor=COL[c], elinewidth=0.8, capsize=0)
        dx, dy, ha = {"FT-NMPC-v2": (4, 1.5, "left"), "EKF-NMPC-v2": (-4, 2.0, "right"), "FT-NMPC-v1": (4, 3.0, "left"),
                      "NMPC-nom": (-4, -4.5, "right"), "PID-35": (-4, 1.5, "right"), "PID-60": (4, 1.5, "left")}[c]
        ax.text(pp.mean() + dx, mm.mean() + dy, LABEL[c], fontsize=6.8, color=INK, ha=ha)
    ax.set_xlim(210, 345); ax.set_xlabel("Production efficiency (USD per reactor hour)"); ax.set_ylabel("Margin per batch (USD)")
    ax.set_xlim(210, 345)
    save(fig, "Fig9_economics")


# Fig 1 overview
def fig1():
    fig, ax = plt.subplots(figsize=(190 * MM, 82 * MM)); ax.set_xlim(0, 190); ax.set_ylim(0, 82); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f3f6fb", ec=COL["FT-NMPC-v2"]):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h - 2.5, title, ha="center", va="top", fontsize=7.4, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 7.6, body, ha="center", va="top", fontsize=6.0, color=INK2, linespacing=1.3)

    def arrow(x1, y1, x2, y2, col=INK2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, shrinkA=0, shrinkB=0))
    box(2, 44, 38, 35, "Pilot reactor (plant)", "600 kg oil, K$_2$CO$_3$\nMeOH:oil 6:1\njacket fed by a TCU\nfitted kinetics\nhidden mismatch\neight fault classes",
        fc="#fbf4ef", ec=COL["EKF-NMPC-v2"])
    box(47, 44, 33, 35, "Measurements", "T, T$_J$, T$_{Jin}$, T$_{cmd}$, Q$_J$\nstirrer power\napparent pH\nload cell\nevery 0.25 min", fc="#f6f6f4", ec=MUTED)
    box(88, 58, 47, 21, "Detection", "residual $\\chi^2$ test\nmodel-free alarms:\nload cell, TCU, stirrer")
    box(88, 25, 47, 27, "Causal estimator", "two-layer GRU\nstates, conversion and\nfault parameters\nEKF as baseline")
    box(143, 25, 45, 54, "Economic NMPC", "30-min horizon\nsolved every 5 min\ninputs: TCU setpoint,\ncatalyst top-up,\nmethanol make-up\n\nStop rule v2:\n$\\hat{X}_E - \\delta \\geq 96.5\\%$\n$\\delta$ = 0.67 or 1.03 point\n5-min confirmation")
    box(2, 2, 186, 11, "Evaluation", "240 fresh scenarios, six controllers, paired tests; Simulink reproduces Python to 4 decimals",
        fc="#ffffff", ec=MUTED)
    arrow(40, 61, 47, 61); arrow(80, 68, 88, 68); arrow(80, 52, 88, 42)
    arrow(135, 68, 143, 68); ax.text(139, 70, "alarm", ha="center", fontsize=6, color=INK2)
    arrow(135, 38, 143, 38); ax.text(139, 40, "$\\hat{x}, \\hat{q}$", ha="center", fontsize=6.5, color=INK2)
    c = COL["FT-NMPC-v2"]
    ax.plot([165.5, 165.5, 21, 21], [25, 19, 19, 44], color=c, lw=1.0)
    arrow(21, 41, 21, 44, col=c)
    ax.text(93, 20.3, "inputs u = [T$_{cmd}$, u$_{cat}$, F$_A$] held for 5 min, and the stop decision", fontsize=6.4, color=c, ha="center")
    save(fig, "Fig1_overview")


if __name__ == "__main__":
    for f in [fig1, fig3, fig4, fig5, fig6, fig7, fig8, fig9]:
        try:
            f(); print("ok", f.__name__, flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc(); print("FAIL", f.__name__)


# Fig 2 kinetic fit (adopted hydroxide model)
def fig2():
    from plant import Params, initial_state, simulate, X_E, preneutralise
    SC = {"catalyst": COL["FT-NMPC-v2"], "ratio": COL["EKF-NMPC-v2"], "time": COL["FT-NMPC-v1"], "temperature": COL["NMPC-nom"]}
    SN = {"catalyst": "catalyst loading", "ratio": "methanol ratio", "time": "reaction time", "temperature": "temperature"}
    fig = plt.figure(figsize=(190 * MM, 72 * MM)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.3)
    ax = fig.add_subplot(gs[0]); panel(ax, "(a)")
    ax.fill_between([60, 102], [57, 99], [63, 105], color="#e9e8e4", lw=0, zorder=0)
    ax.plot([60, 102], [60, 102], color=INK, lw=0.7, zorder=1)
    out = {}
    for case, mk, grp in [("lit_veg", "o", "soybean/cottonseed"), ("lit_jco_kH", "^", "Jatropha")]:
        d = pd.read_csv(f"{ROOT}/model/fit_phase1_{case}_pred.csv")
        rm = np.sqrt(((d["mean"] - d["pred"]) ** 2).mean()); out[grp] = (len(d), rm)
        for s_, g in d.groupby("series"):
            ax.scatter(g["mean"], g["pred"], marker=mk, s=16, color=SC[s_], edgecolor="white", lw=0.4, zorder=3)
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], ls="", marker="s", ms=4, color=SC[k], label=SN[k]) for k in SC]
    lg = ax.legend(handles=h1, loc="upper left", fontsize=6.2, frameon=False, handletextpad=0.1, borderaxespad=0.2, title="series", title_fontsize=6.2)
    lg._legend_box.align = "left"; ax.add_artist(lg)
    h2 = [Line2D([], [], ls="", marker="o", ms=4, color=MUTED, label=f"vegetable oils (n {out['soybean/cottonseed'][0]}, RMSE {out['soybean/cottonseed'][1]:.1f})"),
          Line2D([], [], ls="", marker="^", ms=4, color=MUTED, label=f"Jatropha (n {out['Jatropha'][0]}, RMSE {out['Jatropha'][1]:.1f})")]
    ax.legend(handles=h2, loc="lower right", bbox_to_anchor=(1.0, 0.06), fontsize=6.2, frameon=False, handletextpad=0.1)
    ax.set_xlim(60, 102); ax.set_ylim(60, 102); ax.set_aspect("equal")
    ax.set_xlabel("Measured FAME yield (%)"); ax.set_ylabel("Model ester conversion (%)")
    ax.text(101, 61, "shaded: ±3 points", fontsize=6.2, color=INK2, ha="right")
    ax2 = fig.add_subplot(gs[1]); panel(ax2, "(b)")
    th = json.load(open(f"{ROOT}/model/fit_phase1_lit_veg.run.json"))["theta"]
    tt = np.linspace(0, 120, 481)
    for T, col, lab in [(60, COL["FT-NMPC-v2"], "60 °C"), (55, COL["NMPC-nom"], "55 °C")]:
        p = Params(a=th["a"], GL_sat=1e-3, c_th=th["c_th"], k_SE=th["k_SE"], Ea_SE=th["Ea_SE"], k_H=th["k_H"], k_N=0.0, k_S=0.0)
        x0 = preneutralise(initial_state(m_oil_g=100, MW_oil=848.3, ratio=6, cat_g=4, AV=0.920, water_wt=0.0, T_C=T))
        s = simulate(x0, p, 120, method="LSODA", rtol=1e-6, atol=1e-10)
        ax2.plot(tt, [100 * X_E(s.sol(t), x0) for t in tt], color=col, lw=1.3, label=f"model, {lab}")
    d = pd.read_csv(f"{ROOT}/model/fit_phase1_lit_veg_pred.csv"); d = d[(d.oil == "CWVO") & (d.cat_wt == 4) & (d.MeOH_ratio == 6)]
    for T, col in [(60, COL["FT-NMPC-v2"]), (55, COL["NMPC-nom"])]:
        g = d[d.T_C == T]; ax2.scatter(g.t_min, g["mean"], s=20, color=col, edgecolor=INK, lw=0.5, zorder=4)
    ax2.scatter([], [], s=20, color="white", edgecolor=INK, lw=0.5, label="measured (two campaigns)")
    ax2.axhline(96.5, color=INK, lw=0.7, ls=":"); ax2.text(122, 96.9, "EN 14214, 96.5 %", fontsize=6.5, color=INK2, ha="right", va="bottom")
    ax2.set_xlim(0, 123); ax2.set_ylim(80, 101); ax2.set_yticks([80, 85, 90, 95, 100]); ax2.set_xlabel("t (min)"); ax2.set_ylabel("Ester conversion (%)")
    ax2.set_title("CWVO, 4 wt% K$_2$CO$_3$, methanol:oil 6:1", fontsize=7.5, loc="left")
    ax2.legend(loc="lower right", fontsize=6.5, frameon=False)
    save(fig, "Fig2_kinetic_fit")
