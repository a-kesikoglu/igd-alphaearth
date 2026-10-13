"""Result figures (greyscale, 10 pt serif, 600 dpi PNG + PDF)."""
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import ROOT, CFG, CLASSES, block_ids, load, DISP

plt.rcParams.update({"font.family": "serif", "font.serif": ["Cambria", "DejaVu Serif"], "font.size": 10,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})
M = ROOT / "outputs/metrics"; FIG = ROOT / "outputs/figures"
STY = {"S2-annual": dict(c="0.6", marker="v", ls=":", mfc="white"),
       "S2-seasonal": dict(c="0.35", marker="s", ls="--", mfc="white"),
       "AE": dict(c="black", marker="o", ls="-", mfc="black"),
       "AE+S2-seasonal": dict(c="black", marker="D", ls="-.", mfc="white")}
TNAME = {"ES_holdout": "Spain, spatial holdout", "EL_transfer": "Greece, external transfer"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------- Figure 1: study area and validation design ----------
gj = json.load(open(ROOT / "data/raw/boundaries_lsib2017.geojson"))
from pyproj import Transformer
tr = Transformer.from_crs("EPSG:4326", CFG["projection"], always_xy=True)


def draw_countries(ax, focus):
    for f in gj["features"]:
        g = f["geometry"]; geoms = g["geometries"] if g["type"] == "GeometryCollection" else [g]
        polys = [pp for gg in geoms if gg["type"] in ("Polygon", "MultiPolygon")
                 for pp in (gg["coordinates"] if gg["type"] == "MultiPolygon" else [gg["coordinates"]])]
        for p in polys:
            x, y = tr.transform(*np.array(p[0]).T)
            ax.fill(np.array(x) / 1e3, np.array(y) / 1e3, fc="0.93" if f["properties"]["country_na"] == focus else "white",
                    ec="0.4", lw=0.4, zorder=0)


split = pd.read_csv(M / "split_spain.csv"); df = load(); el = df[df.nuts0 == "EL"]
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.25), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]; draw_countries(ax, "Spain")
L = CFG["experiment"]["holdout_block_km"]
for b in split.loc[split.split == "test", "block50"].unique():
    bx, by = divmod(int(b), 100000)
    ax.add_patch(Rectangle((bx * L, by * L), L, L, fc="0.82", ec="black", lw=0.5, zorder=1))
for s, kw in [("train_pool", dict(s=1.0, c="0.45", lw=0)), ("excluded_buffer", dict(s=6, facecolors="white", edgecolors="black", lw=0.5)),
              ("test", dict(s=4, marker="x", c="black", lw=0.5))]:
    d = split[split.split == s]; ax.scatter(d.x / 1e3, d.y / 1e3, zorder=2, label={"train_pool": "Training pool", "excluded_buffer": "Removed by 5 km buffer", "test": "Locked test sample"}[s], **kw)
ax.add_patch(Rectangle((0, 0), 0, 0, fc="0.82", ec="black", lw=0.5, label="Locked test block (50 km)"))
ax.set_xlim(2650, 3850); ax.set_ylim(1500, 2500); ax.set_aspect("equal")
ax.set_title("(a) Spain", loc="left", fontsize=12)
ax2 = axs[1]; draw_countries(ax2, "Greece")
ax2.scatter(el.x / 1e3, el.y / 1e3, s=1.5, marker="^", c="black", lw=0, zorder=2, label="Greek transfer sample")
ax2.set_xlim(5200, 6050); ax2.set_ylim(1500, 2280); ax2.set_aspect("equal")
ax2.set_title("(b) Greece", loc="left", fontsize=12)
for a in axs:
    a.set_xlabel("Easting (km)", fontsize=11.5); a.tick_params(labelsize=10.5)
    x0, x1 = a.get_xlim(); y0, y1 = a.get_ylim()
    a.plot([x0 + 0.05 * (x1 - x0), x0 + 0.05 * (x1 - x0) + 200], [y0 + 0.05 * (y1 - y0)] * 2, c="black", lw=1.5)
    a.text(x0 + 0.05 * (x1 - x0) + 100, y0 + 0.07 * (y1 - y0), "200 km", ha="center", va="bottom", fontsize=10.5)
    a.annotate("N", xy=(x0 + 0.06 * (x1 - x0), y1 - 0.06 * (y1 - y0)), xytext=(x0 + 0.06 * (x1 - x0), y1 - 0.18 * (y1 - y0)),
               ha="center", fontsize=11, arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black"))
axs[0].set_ylabel("Northing (km)", fontsize=11.5)
h1, l1 = axs[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=10.5, markerscale=3, bbox_to_anchor=(0.5, -0.13), handletextpad=0.3, columnspacing=1.2)
save(fig, "fig1_study_area")

print("figure 1 ok")
