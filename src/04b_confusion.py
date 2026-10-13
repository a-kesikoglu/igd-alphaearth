"""Row-normalised confusion matrices (RF, n = 100, summed over replicates)."""
import numpy as np, pandas as pd
from common import ROOT, CLASSES, confusion
z = np.load(ROOT / "outputs/metrics/e1_predictions.npz"); keys = [tuple(k.split("|")) for k in z["keys"]]
out = []
for t, y in [("ES_holdout", z["y_es_test"]), ("EL_transfer", z["y_el"])]:
    for rep in ["S2-seasonal", "AE"]:
        cm = sum(confusion(y, z[f"p{i}"].astype(int)) for i, k in enumerate(keys) if k[0] == "100" and k[2] == "RF" and k[3] == rep and k[4] == t)
        cmn = cm / cm.sum(1, keepdims=True)
        d = pd.DataFrame(cmn.round(3), index=CLASSES, columns=CLASSES); d.insert(0, "test", t); d.insert(1, "repr", rep); out.append(d)
res = pd.concat(out); res.to_csv(ROOT / "outputs/metrics/confusion_rf_n100.csv"); print(res.to_string())
