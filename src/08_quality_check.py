"""Manuscript checks: style patterns, hyphenation, citations and numbers traced to result files."""
import re, glob
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
md = (ROOT / "manuscript/manuscript.md").read_text(encoding="utf-8")
body = md[md.index("@ABSTRACT"):md.index("### Acknowledgements")]
text = "\n".join(l for l in body.splitlines() if not l.startswith(("!EQ", "!TAB", "!FIG")))
rep = []
rep.append(f"em dashes: {text.count(chr(0x2014))}")
banned = ["Overall", "In conclusion", "Finally", "Notably", "crucial", "delve", "leverag", "groundbreaking", "state-of-the-art",
          "state of the art", "not only", "(i)", "(ii)", "Moreover", "Furthermore", "In summary", "significantly", "novel",
          "unprecedented", "revolution", "paradigm", "landscape of", "It is worth", "Importantly", "Additionally"]
for b in banned:
    n = len(re.findall(re.escape(b), text))
    if n: rep.append(f"pattern '{b}': {n}")
allowed = {"Sentinel-2", "Sentinel-1", "Level-2A", "cross-validation", "Cloud", "Land", "Use/Cover"}
hy = sorted(set(w for w in re.findall(r"\b[\w]+-[\w]+\b", text) if w not in allowed))
rep.append("hyphenated tokens: " + ", ".join(hy))
paras = [p for p in text.split("\n") if p and not p.startswith(("#", "@"))]
for p in paras:
    first = re.split(r"(?<=[.!?])\s", p)[0]
    if len(first.split()) < 8: rep.append(f"short opening sentence: {first}")
# numbers: every 0.xxx value in text must exist in a result file
vals = set()
for f in glob.glob(str(ROOT / "outputs/**/*.csv"), recursive=True):
    for v in re.findall(r"-?\d+\.\d+", open(f, encoding="utf-8").read()):
        x = float(v); vals.update({f"{x:.3f}", f"{abs(x):.3f}"})
s = pd.read_csv(ROOT / "outputs/metrics/e1_summary.csv")
num = re.findall(r"(?<![\d.])0\.\d{3}(?!\d)", text)
missing = sorted(set(n for n in num if n not in vals))
rep.append(f"three-decimal numbers in text: {len(num)}, not found verbatim in result files: {missing}")
out = "\n".join(rep); (ROOT / "reports/manuscript_audit.md").write_text("# Manuscript audit\n\n" + out, encoding="utf-8"); print(out)
