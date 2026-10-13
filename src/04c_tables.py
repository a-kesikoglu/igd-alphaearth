"""Manuscript tables (Tables 1-2) and supplementary tables (S1-S9), generated from result files only."""
import json
import pandas as pd
from common import ROOT, CLASSES, DISP
M = ROOT / "outputs/metrics"; T = ROOT / "outputs/tables"; R = ROOT / "reports"
sp = pd.read_csv(M / "split_spain.csv"); el = json.load(open(R / "split_audit.json"))["greece_class_counts"]
code = dict(zip(CLASSES, "BCDEF"))
fmt = lambda v: f"{v:,}"
cnt = sp.groupby(["label", "split"]).size().unstack(fill_value=0)
t1 = pd.DataFrame({"Class (LUCAS group)": [f"{c} ({code[c]})" for c in CLASSES],
                   "Spain training pool": [cnt.loc[c, "train_pool"] for c in CLASSES],
                   "Spain locked test": [cnt.loc[c, "test"] for c in CLASSES],
                   "Spain buffer excluded": [cnt.loc[c, "excluded_buffer"] for c in CLASSES],
                   "Greece external test": [el[c] for c in CLASSES]})
t1.loc[len(t1)] = ["Total"] + [t1.iloc[:, j].sum() for j in range(1, 5)]
for c in t1.columns[1:]:
    t1[c] = t1[c].map(lambda v: fmt(int(v)))
t1.to_csv(T / "table1_samples.csv", index=False)

agg = pd.read_csv(M / "e1_summary.csv", dtype={"budget": str})
B = ["10", "25", "50", "100", "all"]
COLS = ["n = 10", "n = 25", "n = 50", "n = 100", "Full pool"]


def score_table(model, reprs, name):
    rows = []
    for t, tn in [("ES_holdout", "Spain"), ("EL_transfer", "Greece")]:
        for r in reprs:
            d = agg[(agg.model == model) & (agg.test == t) & (agg.repr == r)].set_index("budget")
            rows.append([tn, DISP[r]] + [f"{d.loc[b, 'mf1_mean']:.3f} ± {d.loc[b, 'mf1_sd']:.3f}" for b in B])
    pd.DataFrame(rows, columns=["Test set", "Representation"] + COLS).to_csv(T / name, index=False)


score_table("RF", ["S2-seasonal", "S1+S2-seasonal", "AE"], "table2_macro_f1.csv")
ALL = ["S2-annual", "S2-seasonal", "S1+S2-seasonal", "S1asc+S2-seasonal", "S1desc+S2-seasonal", "AE", "AE-unit", "AE+S2-seasonal"]
score_table("RF", ALL, "tableS2_rf_all.csv")
score_table("LR", ALL, "tableS3_lr_all.csv")

dl = pd.read_csv(M / "e1_paired_deltas.csv", dtype={"budget": str})
ci = lambda d, b: f"{d.loc[b, 'delta_mean']:.3f} ({d.loc[b, 'ci2_lo']:.3f}, {d.loc[b, 'ci2_hi']:.3f})".replace("-", "−").replace("−0.000", "0.000")
rows = []
for m in ["RF", "LR"]:
    for base in ["S2-seasonal", "S1+S2-seasonal"]:
        for t, tn in [("ES_holdout", "Spain"), ("EL_transfer", "Greece")]:
            d = dl[(dl.test == t) & (dl.model == m) & (dl.a == "AE") & (dl.b == base)].set_index("budget")
            rows.append([m, f"AE − {DISP[base]}", tn] + [ci(d, b) for b in B])
pd.DataFrame(rows, columns=["Classifier", "Comparison", "Test set"] + COLS).to_csv(T / "tableS4_paired_delta.csv", index=False)

rows = []
for a, base in [("S1+S2-seasonal", "S2-seasonal"), ("S1asc+S2-seasonal", "S1+S2-seasonal"),
                ("S1desc+S2-seasonal", "S1+S2-seasonal"), ("AE-unit", "AE")]:
    for t, tn in [("ES_holdout", "Spain"), ("EL_transfer", "Greece")]:
        d = dl[(dl.test == t) & (dl.model == "RF") & (dl.a == a) & (dl.b == base)].set_index("budget")
        rows.append([f"{DISP[a]} − {DISP[base]}", tn] + [ci(d, b) for b in B])
pd.DataFrame(rows, columns=["Comparison (RF)", "Test set"] + COLS).to_csv(T / "tableS5_sensitivity.csv", index=False)

cb = pd.read_csv(M / "e1_cross_budget.csv")
nm = lambda s: s.replace("S1+S2-seasonal", "S1 + S2 seasonal").replace("S2-seasonal", "S2 seasonal").replace("@", ", n = ")
cb["Comparison"] = cb.a.map(nm) + " vs " + cb.b.map(nm)
cb["Test set"] = cb.test.map({"ES_holdout": "Spain", "EL_transfer": "Greece"})
cb["Difference (95% interval)"] = cb.apply(lambda r: f"{r.delta:.3f} ({r.ci2_lo:.3f}, {r.ci2_hi:.3f})".replace("-", "−").replace("−0.000", "0.000"), axis=1)
cb[["Test set", "model", "Comparison", "Difference (95% interval)"]].rename(columns={"model": "Classifier"}).to_csv(
    T / "tableS6_cross_budget.csv", index=False)

sel = pd.read_csv(R / "selection_audit.csv"); sel.columns = ["Selection step", "Total", "Spain", "Greece"]
for c in ["Total", "Spain", "Greece"]:
    sel[c] = sel[c].map(lambda v: fmt(int(v)))
sel.to_csv(T / "tableS1_selection.csv", index=False)

lad = pd.read_csv(M / "e2_ladder_summary.csv")
l = lad[lad.repr == "AE"].groupby("block_km")[["n_test", "n_pool", "nn_km"]].median().reset_index()
l["Protocol"] = l.block_km.map(lambda v: "Random" if v == 0 else f"{v} km blocks")
for c in ["n_test", "n_pool"]:
    l[c] = l[c].map(lambda v: fmt(int(round(v))))
l["nn_km"] = l.nn_km.round(1)
l[["Protocol", "n_test", "n_pool", "nn_km"]].rename(columns={"n_test": "Median test size", "n_pool": "Median training pool",
    "nn_km": "Median distance to nearest training sample (km)"}).to_csv(T / "tableS7_ladder.csv", index=False)

fs = pd.read_csv(M / "feature_shift.csv")
fs["Representation"] = fs["repr"].map(DISP); fs["Test set"] = fs.test.map({"ES_holdout": "Spain", "EL_transfer": "Greece"})
fs["Median DI"] = fs.median_DI.map(lambda v: f"{v:.2f}"); fs["Share with DI > 2"] = fs.share_DI_gt_2.map(lambda v: f"{100 * v:.1f}%")
fs[["Representation", "Test set", "Median DI", "Share with DI > 2"]].to_csv(T / "tableS8_dissimilarity.csv", index=False)

s1 = pd.read_csv(R / "s1_scene_counts.csv").pivot(index="quarter", columns="orbit", values="scenes").reset_index()
s1.columns = ["Quarter", "Ascending scenes", "Descending scenes"]
s1.to_csv(T / "tableS9_s1_scenes.csv", index=False)
for f in ["table2_macro_f1", "tableS5_sensitivity", "tableS6_cross_budget"]:
    print(open(T / f"{f}.csv", encoding="utf-8").read())
