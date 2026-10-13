"""Feature-space shift between the Spanish training pool and the two test sets.
For each representation, features are standardised with the training-pool mean and SD, and the distance of
every test polygon to its nearest training-pool polygon is divided by the median nearest-neighbour distance
within the pool (dissimilarity index in the spirit of the area of applicability, Meyer & Pebesma)."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import ROOT, FEATS, CLASSES, load
df = load(); sp = pd.read_csv(ROOT / "outputs/metrics/split_spain.csv")
es = df[df.nuts0 == "ES"].merge(sp[["id", "split"]], on="id"); el = df[df.nuts0 == "EL"]
pool, test = es[es.split == "train_pool"], es[es.split == "test"]
rows = []
for rep in ["S2-annual", "S2-seasonal", "S1+S2-seasonal", "AE"]:
    cols = FEATS[rep]; mu, sd = pool[cols].mean(), pool[cols].std()
    P = ((pool[cols] - mu) / sd).values; t = cKDTree(P)
    d_in = np.median(t.query(P, k=2)[0][:, 1])
    for name, d in [("ES_holdout", test), ("EL_transfer", el)]:
        dd = t.query(((d[cols] - mu) / sd).values)[0] / d_in
        rows.append({"repr": rep, "test": name, "median_DI": np.median(dd), "share_DI_gt_2": (dd > 2).mean()})
out = pd.DataFrame(rows); out.to_csv(ROOT / "outputs/metrics/feature_shift.csv", index=False); print(out.round(3).to_string())
pr = pd.DataFrame({"pool": pool.label.value_counts(normalize=True), "ES_test": test.label.value_counts(normalize=True),
                   "EL": el.label.value_counts(normalize=True)}).loc[CLASSES]
pr.to_csv(ROOT / "outputs/metrics/class_shares.csv"); print(pr.round(3))
