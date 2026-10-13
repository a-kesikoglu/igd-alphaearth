"""Long-format per-sample prediction table for independent verification.
Columns: run_id, test_set, sample_id, point_id, block_id, true_label, prediction, representation, classifier,
budget, repetition, rf_seed, sampling_seed."""
import numpy as np, pandas as pd
from common import ROOT, CLASSES, load, block_ids, CFG

M = ROOT / "outputs/metrics"
z = np.load(M / "e1_predictions.npz")
keys = [tuple(k.split("|")) for k in z["keys"]]
df = load(); sp = pd.read_csv(M / "split_spain.csv")
es_test = sp[sp.split == "test"].reset_index(drop=True)          # same order as in 03_experiments
el = df[df.nuts0 == "EL"].reset_index(drop=True)
meta = {"ES_holdout": (es_test.id.values, es_test.point_id.values, es_test.block50.values, z["y_es_test"]),
        "EL_transfer": (el.id.values, el.point_id.values, block_ids(el.x, el.y, CFG["experiment"]["holdout_block_km"]), z["y_el"])}
assert (np.array([CLASSES.index(c) for c in es_test.label]) == z["y_es_test"]).all(), "order mismatch"
parts = []
for i, (budget, rep, model, repr_, test) in enumerate(keys):
    ids, pids, blk, y = meta[test]
    seed = 10_000 + 97 * int(rep) + (0 if budget == "all" else int(budget))
    parts.append(pd.DataFrame({"run_id": f"{budget}|{rep}|{model}|{repr_}", "test_set": test, "sample_id": ids,
                               "point_id": pids, "block_id": blk, "true_label": np.array(CLASSES)[y],
                               "prediction": np.array(CLASSES)[z[f"p{i}"].astype(int)], "representation": repr_,
                               "classifier": model, "budget": budget, "repetition": int(rep), "rf_seed": int(rep),
                               "sampling_seed": seed}))
out = pd.concat(parts, ignore_index=True)
out.to_parquet(M / "predictions_long.parquet", index=False)
print(len(out), "rows;", out.run_id.nunique(), "runs")
