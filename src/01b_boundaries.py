"""Download simplified national boundaries (US DoS LSIB 2017, public domain) for maps."""
import json
import ee
from common import ROOT, CFG

ee.Initialize(project=CFG["ee_project"])
fc = (ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
      .filter(ee.Filter.inList("country_na", ["Spain", "Greece", "Portugal", "France", "Andorra", "Morocco",
                                              "Albania", "North Macedonia", "Macedonia", "Bulgaria", "Turkey", "Italy"]))
      .map(lambda f: f.simplify(500)).select(["country_na"]))
json.dump(fc.getInfo(), open(ROOT / "data/raw/boundaries_lsib2017.geojson", "w"))
print("ok")
