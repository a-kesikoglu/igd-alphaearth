"""Figure 2: experimental workflow in three columns (black and white)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import ROOT

plt.rcParams.update({"font.family": "serif", "font.serif": ["Cambria", "DejaVu Serif"], "font.size": 10})
fig, ax = plt.subplots(figsize=(6.5, 3.4)); ax.set_xlim(-1, 101); ax.set_ylim(5, 66); ax.axis("off")


def box(x, y, w, h, text, fill="white", bold=False, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fill, ec="black", lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.15)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black"))


cols = [(0, "Data and features"), (35, "Training protocol"), (70, "Evaluation")]
for x, t in cols:
    ax.text(x + 15, 63.5, t, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.plot([x, x + 30], [61, 61], c="black", lw=0.6)
# data column
box(0, 47, 30, 11, "LUCAS 2018 Copernicus\nin situ polygons, 5 classes\nSpain 8,359 | Greece 1,785", "#e8e8e8")
box(0, 31, 30, 12, "Polygon means, 10 m\nS2 annual (10) | S2 seasonal (40)\nS1 + S2 seasonal (48) | AE (64)\nAE + S2 seasonal (104)")
box(0, 15, 30, 12, "Inputs 2018\nSentinel-2 L2A, Cloud Score+\nSentinel-1 GRD | AlphaEarth", "#e8e8e8")
arrow(15, 46.4, 15, 43.6); arrow(15, 27.6, 15, 30.4)
# training column
box(35, 47, 30, 11, "Spain: 62 of 248 blocks (50 km)\nlocked as test set, 5 km buffer\ntraining pool 6,129")
box(35, 31, 30, 12, "Labels per class: 10, 25, 50, 100\n20 repetitions + full pool\nsame polygon IDs for all inputs")
box(35, 15, 30, 12, "Random Forest (300 trees)\nmultinomial logistic regression\nfixed hyperparameters")
arrow(30.7, 52.5, 34.3, 52.5); arrow(30.7, 37, 34.3, 37); arrow(50, 46.4, 50, 43.6); arrow(50, 30.4, 50, 27.6)
# evaluation column
box(70, 47, 30, 11, "Spain locked holdout (2,049)\nGreece external test (1,785)")
box(70, 31, 30, 12, "Macro F1, OA, BA, class F1\npaired differences\ntest block cluster bootstrap")
box(70, 15, 30, 12, "Protocol ladder (Spain)\nrandom, 10, 25, 50, 100 km\nblock holdouts")
arrow(65.7, 21, 69.3, 21); arrow(65.7, 52.5, 69.3, 52.5); arrow(85, 46.4, 85, 43.6)
ax.text(50, 8.5, "Test sets are never used for training, tuning or feature selection", ha="center", fontsize=7.8, style="italic")
fig.savefig(ROOT / "outputs/figures/fig2_workflow.png", dpi=600, bbox_inches="tight")
fig.savefig(ROOT / "outputs/figures/fig2_workflow.pdf", bbox_inches="tight")
print("ok")
