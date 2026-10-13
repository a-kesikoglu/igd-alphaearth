"""Shared helpers: feature sets, spatial blocks, classifiers and metrics."""
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/config.yaml"))
BANDS = CFG["features"]["s2_bands"]
CLASSES = list(CFG["reference"]["classes"].values())
AE = [f"A{i:02d}" for i in range(64)]
S2A = [f"S2a_{b}" for b in BANDS]
S2S = [f"S2q{q}_{b}" for q in range(1, 5) for b in BANDS]
S1S = [f"S1q{q}_{p}" for q in range(1, 5) for p in ("VV", "VH")]
AEU = [f"U{i:02d}" for i in range(64)]
S1AS = [f"S1Aq{q}_{p}" for q in range(1, 5) for p in ("VV", "VH")]
S1DS = [f"S1Dq{q}_{p}" for q in range(1, 5) for p in ("VV", "VH")]   # polygon-mean AE re-normalised to unit length
DISP = {"S2-annual": "S2 annual", "S2-seasonal": "S2 seasonal", "AE": "AE", "AE+S2-seasonal": "AE + S2 seasonal",
        "S1+S2-seasonal": "S1 + S2 seasonal", "AE-unit": "AE (unit norm)",
        "S1asc+S2-seasonal": "S1 ascending + S2 seasonal", "S1desc+S2-seasonal": "S1 descending + S2 seasonal"}
FEATS = {"S2-annual": S2A, "S2-seasonal": S2S, "AE": AE, "AE+S2-seasonal": AE + S2S,
         "S1+S2-seasonal": S1S + S2S, "AE-unit": AEU,
         "S1asc+S2-seasonal": S1AS + S2S, "S1desc+S2-seasonal": S1DS + S2S}


def load():
    df = pd.read_parquet(ROOT / "data/processed/samples.parquet")
    df["cls"] = df["label"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    a = df[AE].values; u = a / np.linalg.norm(a, axis=1, keepdims=True)
    df[AEU] = u
    return df.reset_index(drop=True)


def block_ids(x, y, L_km):
    """Deterministic square block index on the EPSG:3035 grid (origin 0,0)."""
    L = L_km * 1000.0
    bx = np.floor(np.asarray(x) / L).astype(np.int64); by = np.floor(np.asarray(y) / L).astype(np.int64)
    return bx * 100000 + by


def make_model(kind, seed):
    if kind == "RF":
        return RandomForestClassifier(random_state=seed, **CFG["experiment"]["rf"])
    p = CFG["experiment"]["lr"]
    return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=p["max_iter"]))


def confusion(y, yhat, K=len(CLASSES)):
    return np.bincount(y * K + yhat, minlength=K * K).reshape(K, K)


def metrics_from_cm(cm):
    """cm[..., true, pred] -> dict of macro-F1, OA, balanced accuracy, per-class F1."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    p_den = cm.sum(-2); r_den = cm.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(p_den > 0, tp / p_den, 0.0)
        rec = np.where(r_den > 0, tp / r_den, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {"macro_f1": f1.mean(-1), "oa": tp.sum(-1) / cm.sum((-2, -1)),
            "bal_acc": rec.mean(-1), "f1": f1}


def sample_budget(pool_idx, y, n, rng):
    """Draw n samples per class (without replacement) from pool indices."""
    out = []
    for c in np.unique(y[pool_idx]):
        cand = pool_idx[y[pool_idx] == c]
        out.append(rng.choice(cand, size=min(n, len(cand)), replace=False))
    return np.concatenate(out)
