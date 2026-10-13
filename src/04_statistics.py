"""Summaries, paired differences and test-block cluster bootstrap intervals.

Bootstrap algorithm (per test set):
  1. For every training repetition r and test block g, build the confusion matrix C[r, g].
  2. Draw B = 2000 vectors of block weights w* ~ Multinomial(G, 1/G) once; the SAME weights are
     used for every representation, classifier and budget (paired resampling).
  3. For each draw b and repetition r: C*[b, r] = sum_g w*[b, g] C[r, g]; macro F1 (and class F1)
     are recomputed from C*[b, r].
  4. Average over the R repetitions (repetitions are not resampled), giving one value per draw.
  5. Differences between representations/budgets are taken draw by draw; 2.5 and 97.5 percentiles
     form the interval (ci_lo, ci_hi; conditional on the sampled training draws).
Two-level variant (ci2_lo, ci2_hi): in addition, the R training repetitions are resampled with replacement
in every draw. Repetition weights are shared by all representations and classifiers of the same budget
(identical training IDs, paired), and drawn independently for different budgets.
"""
import numpy as np, pandas as pd
from common import ROOT, CFG, CLASSES, metrics_from_cm

OUT = ROOT / "outputs/metrics"; TAB = ROOT / "outputs/tables"; TAB.mkdir(parents=True, exist_ok=True)
K = len(CLASSES); B = CFG["experiment"]["bootstrap"]
runs = pd.read_csv(OUT / "e1_runs.csv", dtype={"budget": str})

# 1. descriptive summaries over training replicates
agg = runs.groupby(["test", "model", "repr", "budget"]).agg(
    n_rep=("macro_f1", "size"), mf1_mean=("macro_f1", "mean"), mf1_sd=("macro_f1", "std"),
    mf1_min=("macro_f1", "min"), mf1_max=("macro_f1", "max"), oa_mean=("oa", "mean"),
    ba_mean=("bal_acc", "mean"), **{f"f1_{c}": (f"f1_{c}", "mean") for c in CLASSES}).reset_index()
agg.to_csv(OUT / "e1_summary.csv", index=False)

# 2. paired differences with cluster bootstrap over test blocks
z = np.load(OUT / "e1_predictions.npz")
keys = [tuple(k.split("|")) for k in z["keys"]]
pred = {k: z[f"p{i}"] for i, k in enumerate(keys)}
sets = {"ES_holdout": (z["y_es_test"], z["block_es_test"]), "EL_transfer": (z["y_el"], z["block_el"])}
rng = np.random.default_rng(7)
W = {}
for t, (y, b) in sets.items():
    ub, inv = np.unique(b, return_inverse=True)
    W[t] = (inv, len(ub), rng.multinomial(len(ub), np.ones(len(ub)) / len(ub), size=B))
CACHE = {}


def boot(t, budget, model, rep):
    """(B,) bootstrap macro F1, (B, K) class F1 (averaged over repetitions) and per-repetition point values."""
    key = (t, budget, model, rep)
    if key in CACHE:
        return CACHE[key]
    y = sets[t][0]; inv, nb, w = W[t]
    reps = sorted({k[1] for k in keys if k[0] == budget and k[2] == model and k[3] == rep and k[4] == t}, key=int)
    cmb = np.zeros((len(reps), nb, K, K))
    for i, r in enumerate(reps):
        np.add.at(cmb[i], (inv, y, pred[(budget, r, model, rep, t)].astype(int)), 1)
    m = metrics_from_cm(np.einsum("bn,rnkl->brkl", w, cmb))
    v = repw(budget, len(reps))
    CACHE[key] = (m["macro_f1"].mean(1), m["f1"].mean(1), metrics_from_cm(cmb.sum(1))["macro_f1"],
                  (v * m["macro_f1"]).sum(1) / len(reps), np.einsum("br,brk->bk", v, m["f1"]) / len(reps))
    return CACHE[key]


VW = {}


def repw(budget, R):
    """Multinomial repetition weights (B, R), one set per budget, shared across representations and test sets."""
    if budget not in VW:
        g = np.random.default_rng(1000 + (0 if budget == "all" else int(budget)))
        VW[budget] = g.multinomial(R, np.ones(R) / R, size=B)
    return VW[budget]


reprs = sorted(set(runs.repr))
pairs = [("AE", "S2-seasonal"), ("AE", "S2-annual"), ("AE+S2-seasonal", "AE"), ("S2-seasonal", "S2-annual")]
if "S1+S2-seasonal" in reprs:
    pairs += [("AE", "S1+S2-seasonal"), ("S1+S2-seasonal", "S2-seasonal")]
if "AE-unit" in reprs:
    pairs += [("AE-unit", "AE")]
if "S1asc+S2-seasonal" in reprs:
    pairs += [("S1asc+S2-seasonal", "S1+S2-seasonal"), ("S1desc+S2-seasonal", "S1+S2-seasonal")]
rows = []
for t in sets:
    for model in ["RF", "LR"]:
        for budget in [str(b) for b in CFG["experiment"]["budgets"]] + ["all"]:
            for a, c in pairs:
                ba, bc = boot(t, budget, model, a), boot(t, budget, model, c)
                d_boot = ba[0] - bc[0]; d_pt = ba[2] - bc[2]; d2 = ba[3] - bc[3]
                rows.append({"test": t, "model": model, "budget": budget, "a": a, "b": c,
                             "delta_mean": d_pt.mean(), "delta_sd_rep": d_pt.std(ddof=1) if len(d_pt) > 1 else np.nan,
                             "share_reps_positive": (d_pt > 0).mean(),
                             "ci_lo": np.percentile(d_boot, 2.5), "ci_hi": np.percentile(d_boot, 97.5),
                             "ci2_lo": np.percentile(d2, 2.5), "ci2_hi": np.percentile(d2, 97.5)})
delta = pd.DataFrame(rows); delta.to_csv(OUT / "e1_paired_deltas.csv", index=False)

# 3. cross-budget comparisons (label saving within the tested budgets)
cb = []
for t, (ra, na), (rb, nb_) in [("ES_holdout", ("AE", "25"), ("S2-seasonal", "100")),
                               ("ES_holdout", ("AE", "25"), ("S1+S2-seasonal", "100")),
                               ("ES_holdout", ("AE", "10"), ("S2-seasonal", "50")),
                               ("EL_transfer", ("AE", "10"), ("S2-seasonal", "100")),
                               ("EL_transfer", ("AE", "25"), ("S1+S2-seasonal", "100"))]:
    if rb not in reprs:
        continue
    for model in ["RF", "LR"]:
        x, yb = boot(t, na, model, ra), boot(t, nb_, model, rb)
        d = x[0] - yb[0]; d2 = x[3] - yb[3]
        cb.append({"test": t, "model": model, "a": f"{ra}@{na}", "b": f"{rb}@{nb_}", "delta": x[2].mean() - yb[2].mean(),
                   "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5), "p_boot_le0": (d <= 0).mean(),
                   "ci2_lo": np.percentile(d2, 2.5), "ci2_hi": np.percentile(d2, 97.5)})
pd.DataFrame(cb).to_csv(OUT / "e1_cross_budget.csv", index=False)

# 4. class-wise differences AE - S2 seasonal (RF)
cr = []
for t in sets:
    for n in ["25", "100"]:
        fa, fs = boot(t, n, "RF", "AE"), boot(t, n, "RF", "S2-seasonal")
        d = fa[1] - fs[1]; d2 = fa[4] - fs[4]
        sup = np.bincount(sets[t][0], minlength=K)
        for k, c in enumerate(CLASSES):
            cr.append({"test": t, "budget": n, "class": c, "support": int(sup[k]),
                       "f1_s2s": agg.query("test==@t and model=='RF' and repr=='S2-seasonal' and budget==@n")[f"f1_{c}"].item(),
                       "f1_ae": agg.query("test==@t and model=='RF' and repr=='AE' and budget==@n")[f"f1_{c}"].item(),
                       "ci_lo": np.percentile(d[:, k], 2.5), "ci_hi": np.percentile(d[:, k], 97.5),
                       "ci2_lo": np.percentile(d2[:, k], 2.5), "ci2_hi": np.percentile(d2[:, k], 97.5)})
cr = pd.DataFrame(cr); cr["delta"] = cr.f1_ae - cr.f1_s2s; cr.to_csv(OUT / "e1_class_deltas.csv", index=False)

# 5. ladder summary
lad = pd.read_csv(OUT / "e2_ladder.csv")
lsum = lad.groupby(["budget", "repr", "block_km"]).agg(mf1_mean=("macro_f1", "mean"), mf1_sd=("macro_f1", "std"),
                                                        nn_km=("median_nn_dist_km", "median"),
                                                        n_test=("n_test", "median"), n_pool=("n_pool", "median")).reset_index()
lsum.to_csv(OUT / "e2_ladder_summary.csv", index=False)
print(delta[(delta.model == "RF") & (delta.a.isin(["AE", "AE-unit"]))].round(3).to_string())
print(pd.DataFrame(cb).round(3).to_string()); print(cr.round(3).to_string())
print(lsum[lsum.repr.isin(["AE", "S2-seasonal"])].round(3).to_string())
