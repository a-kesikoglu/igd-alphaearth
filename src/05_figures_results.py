"""Main result figures (2, 3) and supplementary figures (S2 confusion, S3 dissimilarity); greyscale, 600 dpi PNG + PDF.
Intervals are two-level bootstrap intervals (test blocks and training draws resampled)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from common import ROOT, CFG, CLASSES, DISP, FEATS, load

plt.rcParams.update({"font.family": "serif", "font.serif": ["Cambria", "DejaVu Serif"], "font.size": 10,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})
M = ROOT / "outputs/metrics"; FIG = ROOT / "outputs/figures"
STY = {"S2-annual": dict(c="0.6", marker="v", ls=":", mfc="white"),
       "S2-seasonal": dict(c="0.4", marker="s", ls="--", mfc="white"),
       "S1+S2-seasonal": dict(c="0.2", marker="D", ls="-.", mfc="white"),
       "AE": dict(c="black", marker="o", ls="-", mfc="black")}
TNAME = {"ES_holdout": "Spain, locked holdout", "EL_transfer": "Greece, external test"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


agg = pd.read_csv(M / "e1_summary.csv", dtype={"budget": str}); dl = pd.read_csv(M / "e1_paired_deltas.csv", dtype={"budget": str})
bud = ["10", "25", "50", "100"]; xp = np.arange(4); XPOOL = 4.8

# ---------- Figure 2: learning curves and paired differences ----------
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.75), gridspec_kw={"width_ratios": [1, 1, 1.05]})
for i, (ax, t) in enumerate(zip(axs[:2], TNAME)):
    for rep, kw in STY.items():
        d = agg[(agg.model == "RF") & (agg.test == t) & (agg.repr == rep)].set_index("budget")
        ax.errorbar(xp, d.loc[bud, "mf1_mean"], yerr=d.loc[bud, "mf1_sd"], color=kw["c"], marker=kw["marker"], ls=kw["ls"],
                    mfc=kw["mfc"], ms=4, lw=0.9, capsize=2, elinewidth=0.6, label=DISP[rep])
        ax.errorbar([XPOOL], [d.loc["all", "mf1_mean"]], yerr=[d.loc["all", "mf1_sd"]], marker=kw["marker"], color=kw["c"],
                    mfc=kw["mfc"], ms=4, ls="none", capsize=2, elinewidth=0.6)
    ax.axvline(4.0, c="0.6", lw=0.5, ls=":")
    ax.set_xticks(list(xp) + [XPOOL]); ax.set_xticklabels(bud + ["Full\npool"], fontsize=8.5)
    ax.set_xlabel("Training labels per class"); ax.set_ylim(0.25, 0.73)
    ax.set_title(f"({'ab'[i]}) {TNAME[t]}", loc="left", fontsize=9.5)
axs[0].set_ylabel("Macro F1"); axs[1].set_yticklabels([])
axs[0].legend(fontsize=8.3, loc="lower right", handlelength=2.0, borderaxespad=0.2)
ax = axs[2]
for t, base, off, kw in [("ES_holdout", "S2-seasonal", -0.15, dict(c="black", marker="s", mfc="black")),
                         ("ES_holdout", "S1+S2-seasonal", -0.05, dict(c="black", marker="D", mfc="black")),
                         ("EL_transfer", "S2-seasonal", 0.05, dict(c="0.4", marker="s", mfc="white")),
                         ("EL_transfer", "S1+S2-seasonal", 0.15, dict(c="0.4", marker="D", mfc="white"))]:
    d = dl[(dl.model == "RF") & (dl.test == t) & (dl.a == "AE") & (dl.b == base)].set_index("budget").loc[bud + ["all"]]
    xx = np.array(list(xp) + [XPOOL]) + off
    ax.errorbar(xx, d.delta_mean, yerr=[d.delta_mean - d.ci2_lo, d.ci2_hi - d.delta_mean], color=kw["c"], marker=kw["marker"],
                mfc=kw["mfc"], ms=3.5, ls="none", capsize=1.5, elinewidth=0.6,
                label=f"{'Spain' if t == 'ES_holdout' else 'Greece'}, vs {DISP[base]}")
ax.axhline(0, c="black", lw=0.7); ax.axvline(4.0, c="0.6", lw=0.5, ls=":")
ax.set_xticks(list(xp) + [XPOOL]); ax.set_xticklabels(bud + ["Full\npool"], fontsize=8.5)
ax.set_xlabel("Training labels per class"); ax.set_ylabel("Δ macro F1 (AE − baseline)")
ax.set_title("(c) Paired difference", loc="left", fontsize=9.5)
ax.legend(fontsize=7.8, loc="upper right", handletextpad=0.2, borderaxespad=0.1); ax.set_ylim(-0.02, 0.27)
fig.tight_layout(w_pad=0.6); save(fig, "fig2_learning_curves")

# ---------- Figure 3: validation sensitivity and class-level differences ----------
lad = pd.read_csv(M / "e2_ladder_summary.csv"); cd = pd.read_csv(M / "e1_class_deltas.csv", dtype={"budget": str})
fig = plt.figure(figsize=(6.7, 2.85))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.55)
ax = fig.add_subplot(gs[0])
for rep in ["AE", "S1+S2-seasonal"]:
    for n, ls in [(25, "--"), (100, "-")]:
        kw = STY[rep]; mfc = "white" if n == 25 else kw["c"]
        d = lad[(lad.budget == n) & (lad.repr == rep)].sort_values("block_km")
        ext = agg[(agg.model == "RF") & (agg.test == "EL_transfer") & (agg.repr == rep) & (agg.budget == str(n))]
        ax.errorbar(range(5), d.mf1_mean, yerr=d.mf1_sd, color=kw["c"], marker=kw["marker"], ls=ls, mfc=mfc, ms=3.8, lw=0.9,
                    capsize=1.5, elinewidth=0.6, label=f"{DISP[rep]}, n = {n}")
        ax.errorbar([5.6], ext.mf1_mean, yerr=ext.mf1_sd, color=kw["c"], marker=kw["marker"], mfc=mfc, ms=3.8, capsize=1.5,
                    elinewidth=0.6)
nn = lad[(lad.budget == 25) & (lad.repr == "AE")].sort_values("block_km").nn_km.values
ax.axvline(4.8, c="0.6", lw=0.5, ls=":")
ax.set_xticks(list(range(5)) + [5.6])
ax.set_xticklabels(["Rand.", "10", "25", "50", "100", "Greece"], fontsize=8)
ax.set_xlabel("Holdout block size (km)", fontsize=8.5); ax.set_ylabel("Macro F1")
ax.set_title("(a) Validation sensitivity", loc="left", fontsize=9.5); ax.legend(fontsize=6.6, loc="lower left", borderaxespad=0.1)
yy = np.arange(len(CLASSES))[::-1]
short = {"Cropland": "Crop", "Woodland": "Wood", "Shrubland": "Shrub", "Grassland": "Grass", "Bare land": "Bare"}
for j, t in enumerate(TNAME):
    ax = fig.add_subplot(gs[j + 1])
    for n, mk, off, fc in [("25", "o", 0.12, "white"), ("100", "s", -0.12, "black")]:
        d = cd[(cd.test == t) & (cd.budget == n)].set_index("class").loc[CLASSES]
        ax.errorbar(d.delta, yy + off, xerr=[d.delta - d.ci2_lo, d.ci2_hi - d.delta], fmt=mk, mfc=fc, mec="black",
                    ecolor="black", ms=3.8, capsize=1.5, elinewidth=0.6, label=f"n = {n}")
    sup = cd[(cd.test == t) & (cd.budget == "100")].set_index("class").loc[CLASSES, "support"]
    ax.axvline(0, c="black", lw=0.7); ax.set_yticks(yy)
    ax.set_yticklabels([f"{short[c]} ({s})" for c, s in zip(CLASSES, sup)], fontsize=7.8)
    if j == 1:
        ax.yaxis.tick_right(); ax.spines["right"].set_visible(True); ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.1, 0.36); ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Δ class F1 (AE − S2 seasonal)", fontsize=8.5)
    ax.set_title(f"({'bc'[j]}) {'Spain' if t == 'ES_holdout' else 'Greece'}", loc="left", fontsize=9.5)
    ax.grid(axis="x", c="0.9", lw=0.5)
    if j == 1:
        ax.legend(fontsize=7, loc="lower right", borderaxespad=0.1, handletextpad=0.2)
save(fig, "fig3_validation_class")

# ---------- Supplementary Figure S2: confusion matrices ----------
cm = pd.read_csv(M / "confusion_rf_n100.csv", index_col=0); ab = ["Cr", "Wo", "Sh", "Gr", "Ba"]
fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.1))
for k, (t, rep) in enumerate([("ES_holdout", "S2-seasonal"), ("ES_holdout", "AE"), ("EL_transfer", "S2-seasonal"), ("EL_transfer", "AE")]):
    ax = axs[k]; m = cm[(cm.test == t) & (cm.repr == rep)][CLASSES].values * 100
    ax.imshow(m, cmap="Greys", vmin=0, vmax=100)
    for i in range(5):
        for jj in range(5):
            ax.text(jj, i, f"{m[i, jj]:.0f}", ha="center", va="center", fontsize=7.5, color="white" if m[i, jj] > 55 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels(ab, fontsize=8); ax.set_yticks(range(5)); ax.set_yticklabels(ab if k == 0 else [], fontsize=8)
    ax.set_title(f"({'abcd'[k]}) {'Spain' if t == 'ES_holdout' else 'Greece'}, {DISP[rep]}", fontsize=8, loc="left")
    ax.set_xlabel("Predicted", fontsize=8)
    for sp_ in ax.spines.values():
        sp_.set_visible(True)
axs[0].set_ylabel("Reference", fontsize=8)
save(fig, "figS2_confusion")

# ---------- Supplementary Figure S3: dissimilarity index distributions ----------
df = load(); sp = pd.read_csv(M / "split_spain.csv")
es = df[df.nuts0 == "ES"].merge(sp[["id", "split"]], on="id"); el = df[df.nuts0 == "EL"]
pool, test = es[es.split == "train_pool"], es[es.split == "test"]
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), sharey=True)
for ax, rep in zip(axs, ["S2-seasonal", "S1+S2-seasonal", "AE"]):
    cols = FEATS[rep]; mu, sd = pool[cols].mean(), pool[cols].std(); P = ((pool[cols] - mu) / sd).values; tr = cKDTree(P)
    din = np.median(tr.query(P, k=2)[0][:, 1])
    for d, lab, kw in [(test, "Spain test", dict(c="black", ls="-")), (el, "Greece", dict(c="0.5", ls="--"))]:
        v = np.sort(tr.query(((d[cols] - mu) / sd).values)[0] / din)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=1, label=lab, **kw)
    ax.axvline(1, c="0.7", lw=0.5); ax.set_xlim(0, 4); ax.set_title(DISP[rep], fontsize=9, loc="left")
    ax.set_xlabel("Dissimilarity index", fontsize=8.5)
axs[0].set_ylabel("Cumulative share", fontsize=8.5); axs[0].legend(fontsize=7.5, loc="lower right")
save(fig, "figS3_dissimilarity")
print("figures ok")
