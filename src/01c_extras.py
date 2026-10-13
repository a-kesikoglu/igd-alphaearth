"""Additional Earth Engine extraction:
1. sequential sample selection counts (catalogue -> analysis set), by country and class group;
2. number of 10 m pixels per polygon (AE grid) and number of valid annual S2 pixels;
3. Sentinel-1 GRD 2018 quarterly median VV and VH backscatter (dB), polygon means.
Outputs: reports/selection_audit.csv, data/raw/features_s1_counts.parquet."""
import time
from concurrent.futures import ThreadPoolExecutor
import ee, pandas as pd
from common import ROOT, CFG

ee.Initialize(project=CFG["ee_project"])
Y = CFG["year"]; R = CFG["reference"]; F = CFG["features"]
start, end = f"{Y}-01-01", f"{Y+1}-01-01"
cty = list(R["countries"].values()); groups = list(R["classes"].keys())

# 1. selection audit -----------------------------------------------------------
cat = ee.FeatureCollection(R["asset"]).map(lambda f: f.set("lc_group", ee.String(f.get("cprn_lc")).slice(0, 1)))
steps = [("Catalogue (EU)", cat)]
s1_ = cat.filter(ee.Filter.inList("nuts0", cty)); steps.append(("Spain and Greece", s1_))
s2_ = s1_.filter(ee.Filter.eq("copernicus_cleaned", True)); steps.append(("Quality screened subset", s2_))
s3_ = s2_.filter(ee.Filter.inList("lc_group", groups)); steps.append(("Classes B to F", s3_))
s4_ = s3_.map(lambda f: f.set("a", f.geometry().area(1))).filter(ee.Filter.gte("a", R["min_polygon_area_m2"]))
steps.append(("Area of at least 400 m2", s4_))
rows = []
for name, fc in steps:
    h = fc.reduceColumns(ee.Reducer.frequencyHistogram().group(0, "c"), ["nuts0", "lc_group"]).getInfo()["groups"]
    total = fc.size().getInfo()
    es = sum(sum(g["histogram"].values()) for g in h if g["c"] == "ES")
    el = sum(sum(g["histogram"].values()) for g in h if g["c"] == "EL")
    rows.append({"step": name, "total": total, "ES": es, "EL": el})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "reports/selection_audit.csv", index=False)

# 2-3. extra features ------------------------------------------------------------
ref = s4_
ae = ee.ImageCollection(F["ae_asset"]).filterDate(start, end).mosaic().select("A00").rename("n_pix")
cs = ee.ImageCollection(F["cs_asset"])
s2 = (ee.ImageCollection(F["s2_asset"]).filterDate(start, end).linkCollection(cs, [F["cs_band"]])
      .map(lambda i: i.select("B4").updateMask(i.select(F["cs_band"]).gte(F["cs_threshold"]))))
s2valid = s2.count().gt(0).selfMask().rename("n_pix_s2")
s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(start, end)
      .filter(ee.Filter.eq("instrumentMode", "IW"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select(["VV", "VH"]))
qs = []
for q in range(4):
    d0 = ee.Date(start).advance(3 * q, "month")
    qs.append(s1.filterDate(d0, d0.advance(3, "month")).median().rename([f"S1q{q+1}_VV", f"S1q{q+1}_VH"]))
s1img = ee.Image.cat(qs)
cnt = ae.addBands(s2valid)


def run(ids):
    fc = ref.filter(ee.Filter.inList("id", ids)).select(["id"])
    a = cnt.reduceRegions(collection=fc, reducer=ee.Reducer.count(), scale=10, tileScale=4)
    b = s1img.reduceRegions(collection=a, reducer=ee.Reducer.mean(), scale=10, tileScale=4)
    for k in range(5):
        try:
            return ee.data.computeFeatures({"expression": b.map(lambda f: f.setGeometry(None)),
                                            "fileFormat": "PANDAS_DATAFRAME"})
        except Exception as e:
            time.sleep(10 * (k + 1)); err = e
    raise err


if __name__ == "__main__":
    ids = pd.read_csv(ROOT / "data/raw/lucas2018_reference.csv")["id"].astype(int).sort_values().tolist()
    batches = [ids[i:i + 400] for i in range(0, len(ids), 400)]
    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(run, batches))
    out = pd.concat(parts, ignore_index=True).drop(columns=["geo"], errors="ignore").drop_duplicates("id")
    out.to_parquet(ROOT / "data/raw/features_s1_counts.parquet", index=False)
    print(len(out), out.isna().sum().sum())
    print(out.describe().T.round(2).to_string())
