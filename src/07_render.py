"""Build DOCX, export PDF with Microsoft Word and write page previews (scratch dir given as argv[1])."""
import subprocess, sys
from pathlib import Path
import pymupdf
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(ROOT / "src/06_build_manuscript.py")], check=True)
docx = ROOT / "manuscript/manuscript.docx"; pdf = docx.with_suffix(".pdf")
ps = (f"$w=New-Object -ComObject Word.Application;$w.Visible=$false;$d=$w.Documents.Open('{docx}',$false,$true);"
      f"$d.SaveAs([ref]'{pdf}',[ref]17);$d.Close($false);$w.Quit()")
subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
d = pymupdf.open(pdf); print("pages:", len(d))
if len(sys.argv) > 1:
    out = Path(sys.argv[1]); dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    ims = []
    for i, p in enumerate(d):
        f = out / f"page{i+1}.png"; p.get_pixmap(dpi=dpi).save(f); ims.append(Image.open(f))
    w, h = ims[0].size; cols = 5; rows = (len(ims) + cols - 1) // cols
    m = Image.new("RGB", (w * cols, h * rows), "white")
    for i, im in enumerate(ims): m.paste(im, ((i % cols) * w, (i // cols) * h))
    m.save(out / "montage.png")
