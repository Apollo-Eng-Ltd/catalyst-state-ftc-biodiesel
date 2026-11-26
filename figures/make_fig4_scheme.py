"""Fig. 4: controller design and application scheme (vector, MATLAB-style colours)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 6.4, "mathtext.fontset": "custom",
                     "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper")
BLUE, ORANGE, GREY, GREEN = "#D6E6F5", "#FBE3D6", "#EDEDED", "#E3EED6"
EDGE, ARROW = "#3a3a3a", "#3a3a3a"

fig = plt.figure(figsize=(6.7, 3.95))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 59); ax.axis("off")


def box(x, y, w, h, text, fc, ec=EDGE, lw=0.7, fs=6.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.25)


def line(pts, dashed=False, head=True, lw=0.9):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=ARROW, lw=lw, ls=(0, (3, 2)) if dashed else "-", solid_capstyle="butt", zorder=1)
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=4,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.45", mutation_scale=6, color=ARROW, lw=0, shrinkA=0, shrinkB=0))


def panel(x0, y0, x1, y1, color, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="none", ec=color, lw=0.8, ls=(0, (4, 2.5)), zorder=0))
    ax.text(x0 + 1.2, y1 - 1.8, title, ha="left", va="center", fontsize=7.6, fontweight="bold")


# panels
panel(1, 8.5, 47.5, 58, "#4DA3E0", "Offline design (Algorithm 1)")
panel(52, 8.5, 99.5, 58, "#F0A060", "Online operation (Algorithm 2)")

# ---- offline column
box(24, 51.5, 42, 3.6, "Kinetic model calibrated on experimental yields", BLUE)
box(24, 45.3, 42, 4.6, "Simulink pilot reactor with 8 fault classes\nand hidden mismatch", BLUE)
box(24, 38.6, 42, 4.6, "Generate batch data (Latin hypercube):\ntraining, calibration, test", BLUE)
line([(24, 49.7), (24, 47.6)]); line([(24, 43.0), (24, 40.9)])
box(10, 29.5, 16, 5, "Residual statistics\n$\\mu$(t), $\\Sigma$(t) and threshold $h$", BLUE)
box(28, 29.5, 15, 5, "Train GRU estimator\nand fault classifier", BLUE)
line([(24, 36.3), (24, 34.3), (10, 34.3), (10, 32.0)]); line([(24, 34.3), (28, 34.3), (28, 32.0)])
box(28, 19.5, 19, 5.4, "Conformal stop margins\n$\\delta_0$ (no alarm), $\\delta_1$ (after alarm)", BLUE)
line([(28, 27.0), (28, 22.2)])
box(42.5, 19.5, 8.6, 7.0, "Unlabelled\nadaptation\nto a new\nfeedstock", BLUE, fs=6.0)
line([(35.5, 29.5), (42.5, 29.5), (42.5, 23.0)], dashed=True)
ax.text(36.2, 30.5, "only for a new feed", fontsize=5.6, ha="left", va="bottom", style="italic")

# ---- frozen box
box(50, 3.7, 44, 3.4, "Frozen: $h$, estimator, classifier, alarm thresholds, $\\delta_0$, $\\delta_1$", GREY)
line([(10, 27.0), (10, 3.7), (28.0, 3.7)])
line([(33, 16.8), (33, 5.4)])
line([(42.5, 16.0), (42.5, 5.4)], dashed=True)

# ---- online loop
box(72, 51.3, 22, 3.6, "Batch reactor (plant)", GREY)
box(79, 44.2, 34, 5.0, "8 measurements every 0.25 min: temperatures,\nTCU set-point, jacket duty, stirrer power, pH, load cell", ORANGE)
line([(72, 49.5), (72, 46.7)])
box(68, 33.5, 17, 5, "Residual detector $D$(t)\nand model-free alarms", ORANGE)
box(87.5, 33.5, 19, 5.6, "GRU catalyst-state\nestimate: conversion,\nactive catalyst, faults", ORANGE, fs=6.2)
line([(79, 41.7), (79, 38.6), (68, 38.6), (68, 36.0)]); line([(79, 38.6), (87.5, 38.6), (87.5, 36.3)])
box(85.3, 25.5, 23.4, 5, "Economic NMPC every 5 min:\ntemperature, catalyst top-up,\nmethanol make-up", ORANGE, fs=6.2)
line([(87.5, 30.7), (87.5, 28.0)])
dc, dw, dh = (83, 17.4), 11.5, 4.3
ax.add_patch(Polygon([(dc[0] - dw, dc[1]), (dc[0], dc[1] + dh), (dc[0] + dw, dc[1]), (dc[0], dc[1] - dh)],
                     closed=True, fc=GREEN, ec=EDGE, lw=0.7, zorder=2))
ax.text(dc[0], dc[1], "Stop rule\n$C_1$, $C_2$, $C_3$ met?", ha="center", va="center", fontsize=6.2, zorder=3)
line([(83, 23.0), (83, dc[1] + dh)])
# alarm widens margin: detector -> upper-left edge of the diamond
ya = dc[1]; xa = dc[0] - dw
line([(64, 31.0), (64, ya), (xa, ya)], dashed=True)
ax.text(64.6, 26.5, "Alarm widens\nmargin", fontsize=5.6, ha="left", va="center", style="italic")
# no: back to the plant
line([(dc[0] + dw, dc[1]), (97.8, dc[1]), (97.8, 51.3), (83, 51.3)])
ax.text(95.0, dc[1] + 0.6, "No", fontsize=6.2, ha="left", va="bottom")
ax.text(85, 52.1, "Apply first move", fontsize=5.8, ha="left", va="bottom", style="italic")
# yes: end batch
box(83, 10.4, 14, 2.6, "End batch", ORANGE, ec="#C0504D", lw=0.9)
line([(83, dc[1] - dh), (83, 11.7)])
ax.text(83.8, 12.9, "Yes", fontsize=6.2, ha="left", va="center")
# frozen components into the online loop
line([(57, 5.4), (57, 33.5), (59.5, 33.5)])

fig.savefig(os.path.join(OUT, "Fig4_scheme.png"), dpi=600)
fig.savefig(os.path.join(OUT, "Fig4_scheme.pdf"))
print("ok")
