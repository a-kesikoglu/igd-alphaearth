"""Additional feature quality checks and Sentinel-1 orbit sensitivity features.
Per polygon (same 10 m reduction grid and weighting as the main extraction):
- clear_q{1..4}: area fraction of the polygon with at least one clear Sentinel-2 observation in quarter q;
- s1n_{asc,desc}_q{1..4}: mean number of Sentinel-1 observations per grid cell in quarter q by orbit direction;
- S1A/S1D quarterly VV, VH medians from ascending-only and descending-only scenes.
Also writes collection-level Sentinel-1 scene counts per quarter and orbit (reports/s1_scene_counts.csv)."""
import time
from concurrent.futures import ThreadPoolExecutor
import ee, pandas as pd
from common import ROOT, CFG

ee.Initialize(project=CFG["ee_project"])
Y = CFG["year"]; F = CFG["features"]; R = CFG["reference"]
start = ee.Date(f"{Y}-01-01")
ref = (ee.FeatureCollection(R["asset"]).filter(ee.Filter.inList("nuts0", list(R["countries"].values())))
       .filter(ee.Filter.eq("copernicus_cleaned", True)))
cs = ee.ImageCollection(F["cs_asset"])
s2 = (ee.ImageCollection(F["s2_asset"]).filterDate(start, start.advance(1, "year")).linkCollection(cs, [F["cs_band"]])
      .map(lambda i: i.select("B4").updateMask(i.select(F["cs_band"]).gte(F["cs_threshold"]))))
s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(start, start.advance(1, "year"))
      .filter(ee.Filter.eq("instrumentMode", "IW"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select(["VV", "VH"]))
bands = []
for q in range(4):
    d0 = start.advance(3 * q, "month"); d1 = d0.advance(3, "month")
    bands.append(s2.filterDate(d0, d1).count().gt(0).unmask(0).rename(f"clear_q{q+1}"))
    for o, tag in [("ASCENDING", "asc"), ("DESCENDING", "desc")]:
        c = s1.filterDate(d0, d1).filter(ee.Filter.eq("orbitProperties_pass", o))
        bands.append(c.select("VV").count().unmask(0).rename(f"s1n_{tag}_q{q+1}"))
        bands.append(c.median().rename([f"S1{tag[0].upper()}q{q+1}_VV", f"S1{tag[0].upper()}q{q+1}_VH"]))
stack = ee.Image.cat(bands)


def run(ids):
    fc = ref.filter(ee.Filter.inList("id", ids)).select(["id"])
    out = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4)
    for k in range(5):
        try:
            return ee.data.computeFeatures({"expression": out.map(lambda f: f.setGeometry(None)),
                                            "fileFormat": "PANDAS_DATAFRAME"})
        except Exception as e:
            time.sleep(10 * (k + 1)); err = e
    raise err


if __name__ == "__main__":
    region = ref.geometry().bounds()
    rows = []
    for q in range(4):
        d0 = start.advance(3 * q, "month")
        for o in ["ASCENDING", "DESCENDING"]:
            n = s1.filterDate(d0, d0.advance(3, "month")).filterBounds(region).filter(ee.Filter.eq("orbitProperties_pass", o)).size().getInfo()
            rows.append({"quarter": q + 1, "orbit": o, "scenes": n})
    pd.DataFrame(rows).to_csv(ROOT / "reports/s1_scene_counts.csv", index=False); print(pd.DataFrame(rows))
    ids = pd.read_csv(ROOT / "data/raw/lucas2018_reference.csv")["id"].astype(int).sort_values().tolist()
    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(run, [ids[i:i + 400] for i in range(0, len(ids), 400)]))
    out = pd.concat(parts, ignore_index=True).drop(columns=["geo"], errors="ignore").drop_duplicates("id")
    out.to_parquet(ROOT / "data/raw/features_qc_s1orbit.parquet", index=False)
    print(len(out)); print(out.describe().T.round(3).to_string())
