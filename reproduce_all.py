"""Reproduce every table, figure and statistic from the archived feature tables.

Usage:  python reproduce_all.py            (offline: starts from data/raw/*.parquet)
        python reproduce_all.py --extract  (also re-extracts features from Google Earth Engine)
        python reproduce_all.py --paper    (also rebuilds the paper DOCX/PDF; needs template and MS Word)
Afterwards, outputs/ and manuscript/ are regenerated and reports/manuscript_audit.md lists any number
in the text that is not found in the result files."""
import subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
steps = (["01_export_features.py", "01b_boundaries.py", "01c_extras.py", "01d_qc_s1.py"] if "--extract" in sys.argv else []) + [
    "02_prepare.py", "03_experiments.py", "04_statistics.py", "04b_confusion.py", "04c_tables.py", "04d_shift.py",
    "05_figures.py", "05_figures_results.py", "05b_workflow_figure.py", "09_export_predictions.py", "10_supplement.py",
    "08_quality_check.py"]
if "--paper" in sys.argv:          # needs the IGD Word template one level up and Microsoft Word
    steps.insert(-1, "07_render.py")
for s in steps:
    print(f"== {s}", flush=True)
    subprocess.run([sys.executable, s], cwd=SRC, check=True)
print("done")
