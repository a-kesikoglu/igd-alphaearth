"""E1: locked spatial holdout (Spain) + external transfer (Greece) learning curves.
E2: validation-protocol ladder (random and 10-100 km block holdouts, Spain).
Writes per-sample predictions and per-run metrics to outputs/metrics."""
import json, time
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import (ROOT, CFG, CLASSES, FEATS, load, block_ids, make_model,
                    confusion, metrics_from_cm, sample_budget)

E = CFG["experiment"]; OUT = ROOT / "outputs/metrics"; OUT.mkdir(parents=True, exist_ok=True)
df = load()
es = df[df.nuts0 == CFG["reference"]["countries"]["train_region"]].reset_index(drop=True)
el = df[df.nuts0 == CFG["reference"]["countries"]["transfer_region"]].reset_index(drop=True)
y_es, y_el = es.cls.values, el.cls.values
X_es = {k: es[v].values for k, v in FEATS.items()}; X_el = {k: el[v].values for k, v in FEATS.items()}
K = len(CLASSES)


def locked_holdout():
    """Select ~25% of 50 km blocks as the locked test set; every class needs >= 30 test samples."""
    b = block_ids(es.x, es.y, E["holdout_block_km"]); ub = np.unique(b)
    rng = np.random.default_rng(E["holdout_seed"])
    for attempt in range(1, 1000):
        test_blocks = rng.choice(ub, size=int(round(E["holdout_fraction"] * len(ub))), replace=False)
        test = np.isin(b, test_blocks)
        if np.bincount(y_es[test], minlength=K).min() >= 30 and np.bincount(y_es[~test], minlength=K).min() >= 150:
            return b, test, attempt
    raise RuntimeError("no admissible holdout")


blocks, test_mask, attempt = locked_holdout()
test_idx = np.where(test_mask)[0]
tree = cKDTree(es[["x", "y"]].values[test_idx])
dist, _ = tree.query(es[["x", "y"]].values[~test_mask])
pool_all = np.where(~test_mask)[0]
pool = pool_all[dist >= E["buffer_km"] * 1000]
split = {"holdout_attempt": attempt, "n_blocks": int(len(np.unique(blocks))),
         "n_test_blocks": int(len(np.unique(blocks[test_mask]))), "n_test": int(len(test_idx)),
         "n_pool_before_buffer": int(len(pool_all)), "n_pool": int(len(pool)),
         "removed_by_buffer": int(len(pool_all) - len(pool)),
         "min_train_test_distance_km": float(dist[dist >= E["buffer_km"] * 1000].min() / 1000),
         "test_class_counts": dict(zip(CLASSES, np.bincount(y_es[test_idx], minlength=K).tolist())),
         "pool_class_counts": dict(zip(CLASSES, np.bincount(y_es[pool], minlength=K).tolist())),
         "greece_class_counts": dict(zip(CLASSES, np.bincount(y_el, minlength=K).tolist()))}
assert not set(blocks[pool]) & set(blocks[test_idx]), "block leakage"
es["split"] = "excluded_buffer"; es.loc[pool, "split"] = "train_pool"; es.loc[test_idx, "split"] = "test"
es[["id", "point_id", "label", "x", "y", "split"]].assign(block50=blocks).to_csv(OUT / "split_spain.csv", index=False)
json.dump(split, open(ROOT / "reports/split_audit.json", "w"), indent=2)
print(json.dumps(split, indent=2))

# split identity files (with hashes) and test block polygons for sharing
import hashlib
from pyproj import Transformer
SP = ROOT / "data/splits"; SP.mkdir(parents=True, exist_ok=True)
files = {"train_pool_spain_ids.csv": es.loc[pool, ["id", "point_id", "label"]],
         "test_spain_ids.csv": es.loc[test_idx, ["id", "point_id", "label"]],
         "buffer_excluded_spain_ids.csv": es.loc[es.split == "excluded_buffer", ["id", "point_id", "label"]],
         "test_greece_ids.csv": el[["id", "point_id", "label"]]}
hashes = {}
for f, d in files.items():
    d.sort_values("id").to_csv(SP / f, index=False); hashes[f] = {"n": len(d), "sha256": hashlib.sha256((SP / f).read_bytes()).hexdigest()}
json.dump({"holdout_seed": E["holdout_seed"], "block_km": E["holdout_block_km"], "buffer_km": E["buffer_km"],
           "grid": "EPSG:3035, origin (0, 0), block id = floor(x/L), floor(y/L)", "files": hashes},
          open(SP / "manifest.json", "w"), indent=2)
inv = Transformer.from_crs(CFG["projection"], "EPSG:4326", always_xy=True); L = E["holdout_block_km"] * 1000
feats = []
for b in np.unique(blocks):
    bx, by = divmod(int(b), 100000)
    ring = [inv.transform(bx * L + dx, by * L + dy) for dx, dy in [(0, 0), (L, 0), (L, L), (0, L), (0, 0)]]
    feats.append({"type": "Feature", "properties": {"block_id": int(b), "test": bool(b in set(blocks[test_idx])),
                  "n_samples": int((blocks == b).sum())}, "geometry": {"type": "Polygon", "coordinates": [ring]}})
json.dump({"type": "FeatureCollection", "features": feats}, open(SP / "validation_blocks_50km.geojson", "w"))

# ---------------- E1 ----------------
assert np.unique(blocks).size > 100, "block grid looks wrong"
rows, preds = [], {}
t0 = time.time()
budgets = E["budgets"] + ["all"]
for n in budgets:
    R = E["replicates"] if n != "all" else 5
    for r in range(R):
        rng = np.random.default_rng(10_000 + 97 * r + (0 if n == "all" else n))
        tr = pool if n == "all" else sample_budget(pool, y_es, n, rng)
        for model in ["RF", "LR"]:
            for rep, X in X_es.items():
                m = make_model(model, seed=r).fit(X[tr], y_es[tr])
                for tset, Xt, yt in [("ES_holdout", X[test_idx], y_es[test_idx]), ("EL_transfer", X_el[rep], y_el)]:
                    yhat = m.predict(Xt)
                    preds[(str(n), r, model, rep, tset)] = yhat.astype(np.int8)
                    mt = metrics_from_cm(confusion(yt, yhat))
                    rows.append({"budget": str(n), "rep": r, "model": model, "repr": rep, "test": tset,
                                 "n_train": len(tr), "macro_f1": mt["macro_f1"], "oa": mt["oa"],
                                 "bal_acc": mt["bal_acc"], **{f"f1_{c}": v for c, v in zip(CLASSES, mt["f1"])}})
    print(f"E1 budget {n} done, {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(OUT / "e1_runs.csv", index=False)
keys = list(preds); np.savez_compressed(OUT / "e1_predictions.npz",
    keys=np.array(["|".join(map(str, k)) for k in keys]), **{f"p{i}": preds[k] for i, k in enumerate(keys)},
    y_es_test=y_es[test_idx], y_el=y_el, block_es_test=blocks[test_idx],
    block_el=block_ids(el.x, el.y, E["holdout_block_km"]))

# ---------------- E2 ladder ----------------
rows = []
for L in E["block_sizes_km"]:
    b = np.arange(len(es)) if L == 0 else block_ids(es.x, es.y, L)
    ub = np.unique(b)
    for r in range(E["ladder_replicates"]):
        rng = np.random.default_rng(50_000 + 31 * r + L)
        while True:
            if L == 0:  # stratified random sample-level split
                te = np.concatenate([rng.choice(np.where(y_es == c)[0], int(round(E["holdout_fraction"] * (y_es == c).sum())), replace=False) for c in range(K)])
                tmask = np.zeros(len(es), bool); tmask[te] = True
            else:
                tmask = np.isin(b, rng.choice(ub, int(round(E["holdout_fraction"] * len(ub))), replace=False))
            if np.bincount(y_es[tmask], minlength=K).min() >= 20 and np.bincount(y_es[~tmask], minlength=K).min() >= max(E["ladder_budgets"]):
                break
        tr_pool = np.where(~tmask)[0]; te_idx = np.where(tmask)[0]
        d, _ = cKDTree(es[["x", "y"]].values[tr_pool]).query(es[["x", "y"]].values[te_idx])
        for n in E["ladder_budgets"]:
            tr = sample_budget(tr_pool, y_es, n, rng)
            for rep, X in X_es.items():
                if rep in ("AE-unit", "S1asc+S2-seasonal", "S1desc+S2-seasonal"):
                    continue
                yhat = make_model("RF", r).fit(X[tr], y_es[tr]).predict(X[te_idx])
                mt = metrics_from_cm(confusion(y_es[te_idx], yhat))
                rows.append({"block_km": L, "rep": r, "budget": n, "repr": rep, "n_test": len(te_idx), "n_pool": len(tr_pool),
                             **{f"test_share_{c}": v for c, v in zip(CLASSES, np.bincount(y_es[te_idx], minlength=K) / len(te_idx))},
                             "median_nn_dist_km": float(np.median(d) / 1000),
                             "macro_f1": mt["macro_f1"], "oa": mt["oa"], "bal_acc": mt["bal_acc"]})
    print(f"E2 L={L} km done, {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(OUT / "e2_ladder.csv", index=False)
print("finished", time.time() - t0)
