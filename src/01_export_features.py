"""Extract LUCAS 2018 Copernicus reference polygons and polygon-mean features
(AlphaEarth 2018 embeddings, Sentinel-2 L2A 2018 annual and quarterly medians)
from Google Earth Engine. Output: data/raw/features_raw.parquet + manifest."""
import json, time, hashlib, datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ee, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
ee.Initialize(project=cfg["ee_project"])
Y = cfg["year"]; F = cfg["features"]; R = cfg["reference"]
start, end = f"{Y}-01-01", f"{Y+1}-01-01"

ref = (ee.FeatureCollection(R["asset"])
       .filter(ee.Filter.inList("nuts0", list(R["countries"].values())))
       .filter(ee.Filter.eq("copernicus_cleaned", R["filter_copernicus_cleaned"]))
       .map(lambda f: f.set("lc_group", ee.String(f.get("cprn_lc")).slice(0, 1)))
       .filter(ee.Filter.inList("lc_group", list(R["classes"].keys())))
       .map(lambda f: f.set("poly_area_m2", f.geometry().area(1)))
       .filter(ee.Filter.gte("poly_area_m2", R["min_polygon_area_m2"])))

keep = ["id", "point_id", "nuts0", "nuts1", "nuts2", "cprn_lc", "cprn_lc_label", "lc1",
        "lc_group", "survey_date", "obs_type", "gps_lat", "gps_long", "th_lat", "th_long",
        "parcel_area_ha", "cprn_impervious_perc", "cprn_lc_same_lc1", "poly_area_m2"]

# Feature images -------------------------------------------------------------
ae_col = ee.ImageCollection(F["ae_asset"]).filterDate(start, end)
ae = ae_col.mosaic().select([f"A{i:02d}" for i in range(64)])

bands = F["s2_bands"]
cs = ee.ImageCollection(F["cs_asset"])
def mask_s2(img):
    m = img.select(F["cs_band"]).gte(F["cs_threshold"])
    return img.select(bands).updateMask(m).divide(10000).copyProperties(img, ["system:time_start"])
s2 = (ee.ImageCollection(F["s2_asset"]).filterDate(start, end)
      .linkCollection(cs, [F["cs_band"]]).map(mask_s2))
s2a = s2.median().rename([f"S2a_{b}" for b in bands])
quarters = []
for q in range(4):
    qs = ee.Date(start).advance(3 * q, "month")
    quarters.append(s2.filterDate(qs, qs.advance(3, "month")).median()
                    .rename([f"S2q{q+1}_{b}" for b in bands]))
nobs = s2.select("B4").count().rename("S2_nobs")
qobs = [s2.filterDate(ee.Date(start).advance(3*q, "month"), ee.Date(start).advance(3*q+3, "month"))
        .select("B4").count().rename(f"S2q{q+1}_nobs") for q in range(4)]
stack = ae.addBands(s2a).addBands(ee.Image.cat(quarters)).addBands(nobs).addBands(ee.Image.cat(qobs))

def run_batch(ids):
    fc = ref.filter(ee.Filter.inList("id", ids))
    out = stack.reduceRegions(collection=fc.select(["id"]), reducer=ee.Reducer.mean(),
                              scale=F["scale_m"], tileScale=4)
    for k in range(5):
        try:
            return ee.data.computeFeatures({"expression": out.map(lambda f: f.setGeometry(None)),
                                            "fileFormat": "PANDAS_DATAFRAME"})
        except Exception as e:  # transient quota / timeout errors
            time.sleep(10 * (k + 1)); err = e
    raise err

if __name__ == "__main__":
    raw = ROOT / "data/raw"; raw.mkdir(parents=True, exist_ok=True)
    # paging of computeFeatures is not order-stable; fetch per country in id-sorted chunks
    chunks = []
    for c in R["countries"].values():
        sub = ref.filter(ee.Filter.eq("nuts0", c)).select(keep).sort("id")
        n = sub.size().getInfo()
        for o in range(0, n, 2000):
            chunks.append(ee.data.computeFeatures({"expression": ee.FeatureCollection(sub.toList(2000, o))
                                                   .map(lambda f: f.setGeometry(None)), "fileFormat": "PANDAS_DATAFRAME"}))
    meta = pd.concat(chunks, ignore_index=True).drop_duplicates("id")
    meta = meta.drop(columns=[c for c in meta.columns if c == "geo"], errors="ignore")
    meta.to_csv(raw / "lucas2018_reference.csv", index=False)
    print("reference polygons:", len(meta), meta.groupby(["nuts0", "lc_group"]).size().to_dict())
    ids = sorted(meta["id"].astype(int).tolist()); bs = F["batch_size"]
    done = None
    if (raw / "features_raw.parquet").exists():  # resume: only extract ids not yet present
        done = pd.read_parquet(raw / "features_raw.parquet").drop_duplicates("id")
        ids = sorted(set(ids) - set(done["id"].astype(int)))
        print("resuming, missing ids:", len(ids))
    batches = [ids[i:i + bs] for i in range(0, len(ids), bs)]
    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(run_batch, batches))
    feat = pd.concat(([done] if done is not None else []) + parts, ignore_index=True).drop(columns=["geo"], errors="ignore")
    feat = feat.drop_duplicates("id")
    feat = feat[feat["id"].isin(meta["id"])]
    feat.to_parquet(raw / "features_raw.parquet", index=False)
    ae_meta = ae_col.filterBounds(ref.geometry().bounds()).aggregate_array("system:index").getInfo()
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "ee_api": ee.__version__,
        "year": Y, "reference_asset": R["asset"], "n_reference": int(len(meta)),
        "n_features": int(len(feat)), "ae_asset": F["ae_asset"], "ae_tiles_used": len(ae_meta),
        "s2_asset": F["s2_asset"], "cloud_mask": f"{F['cs_asset']} {F['cs_band']}>={F['cs_threshold']}",
        "reducer": "polygon mean at 10 m", "sha256_features": hashlib.sha256(
            open(raw / "features_raw.parquet", "rb").read()).hexdigest()}
    json.dump(manifest, open(ROOT / "data/manifests/datasets.json", "w"), indent=2)
    print(json.dumps(manifest, indent=2))
