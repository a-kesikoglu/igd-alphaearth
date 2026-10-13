"""Merge reference labels and features, apply QC, project to EPSG:3035 and
write data/processed/samples.parquet plus reports/dataset_audit.md."""
from pathlib import Path
import numpy as np, pandas as pd, yaml
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
bands = cfg["features"]["s2_bands"]
AE = [f"A{i:02d}" for i in range(64)]
S2A = [f"S2a_{b}" for b in bands]
S2S = [f"S2q{q}_{b}" for q in range(1, 5) for b in bands]

ref = pd.read_csv(ROOT / "data/raw/lucas2018_reference.csv")
feat = pd.read_parquet(ROOT / "data/raw/features_raw.parquet")
df = ref.merge(feat, on="id", how="inner")
xtra = ROOT / "data/raw/features_s1_counts.parquet"
if xtra.exists():   # Sentinel-1 quarterly backscatter and pixel counts
    df = df.merge(pd.read_parquet(xtra), on="id", how="left")
S1S = [f"S1q{q}_{p}" for q in range(1, 5) for p in ("VV", "VH")]
qc = ROOT / "data/raw/features_qc_s1orbit.parquet"
if qc.exists():   # quarterly clear coverage, S1 observation counts and orbit-specific S1 features
    df = df.merge(pd.read_parquet(qc), on="id", how="left")
df["label"] = df["lc_group"].map(cfg["reference"]["classes"])

# coordinates: GPS position when valid, otherwise theoretical LUCAS grid point
ok = df["gps_lat"].between(-90, 90) & df["gps_long"].between(-180, 180)
lat = np.where(ok, df["gps_lat"], df["th_lat"]); lon = np.where(ok, df["gps_long"], df["th_long"])
tr = Transformer.from_crs("EPSG:4326", cfg["projection"], always_xy=True)
df["x"], df["y"] = tr.transform(lon, lat)
df["lat"], df["lon"] = lat, lon

lines = ["# Dataset audit", "", f"Reference polygons after GEE filters: {len(ref)}; with features: {len(df)}", ""]
miss = {"AE": df[AE].isna().any(axis=1), "S2-annual": df[S2A].isna().any(axis=1),
        "S2-seasonal": df[S2S].isna().any(axis=1)}
if "S1q1_VV" in df:
    lines.append(f"- rows with missing S1 seasonal: {int(df[S1S].isna().any(axis=1).sum())} (not used to filter; see S1 experiment)")
for k, v in miss.items():
    lines.append(f"- rows with missing {k}: {int(v.sum())}")
valid = ~(miss["AE"] | miss["S2-annual"] | miss["S2-seasonal"])
lines.append(f"- common valid intersection: {int(valid.sum())} of {len(df)}")
drop = df[~valid].groupby(["nuts0", "label"]).size()
lines += ["", "Dropped rows by country/class:", "", drop.to_string() if len(drop) else "none", ""]
df = df[valid].copy()

norm = np.linalg.norm(df[AE].values, axis=1)
lines.append(f"AE polygon-mean vector norm: median {np.median(norm):.3f}, 1st pct {np.percentile(norm,1):.3f}, 99th pct {np.percentile(norm,99):.3f}")
for c in ["S2_nobs"] + [f"S2q{q}_nobs" for q in range(1, 5)]:
    lines.append(f"{c}: median {df[c].median():.1f}, min {df[c].min():.1f}")
s2v = df[S2A].values
lines.append(f"S2-annual reflectance range: {s2v.min():.4f} to {s2v.max():.4f}")
if "n_pix" in df:
    lines.append(f"10 m pixels per polygon: min {df.n_pix.min():.0f}, 1st pct {df.n_pix.quantile(.01):.0f}, median {df.n_pix.median():.0f}, max {df.n_pix.max():.0f}")
    lines.append(f"pixels with >=1 clear S2 observation per polygon: min {df.n_pix_s2.min():.0f}, median {df.n_pix_s2.median():.0f}; share of polygons fully clear {(df.n_pix_s2 >= df.n_pix).mean():.3f}")
if "clear_q1" in df:
    for q in range(1, 5):
        lines.append(f"quarter {q}: min area fraction with >=1 clear S2 observation {df[f'clear_q{q}'].min():.3f}; "
                     f"S1 observations per cell, ascending min {df[f's1n_asc_q{q}'].min():.1f} median {df[f's1n_asc_q{q}'].median():.0f}, "
                     f"descending min {df[f's1n_desc_q{q}'].min():.1f} median {df[f's1n_desc_q{q}'].median():.0f}")
lines.append(f"Duplicate point_id: {int(df['point_id'].duplicated().sum())}")
lines += ["", "Final samples by country and class:", "", df.groupby(["nuts0", "label"]).size().unstack(0).to_string()]
lines += ["", "Survey months (ES+EL):", "", pd.to_datetime(df["survey_date"], format="%d/%m/%y").dt.month.value_counts().sort_index().to_string()]
(ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
df.to_parquet(ROOT / "data/processed/samples.parquet", index=False)
open(ROOT / "reports/dataset_audit.md", "w").write("\n".join(lines))
print("\n".join(lines))
