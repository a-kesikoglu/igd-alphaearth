"""Build the IGD 2026 manuscript (DOCX) from manuscript/manuscript.md on the official template.

Markup in manuscript.md:
  @TITLE, @CITE, @ABSTRACT, @KEYWORDS k1; k2      front matter (title table)
  # Heading           first-level heading (template Heading 2)
  ## Heading          second-level heading (bold, left aligned)
  ### Heading         declaration heading (bold)
  !FIG path | width_cm | caption
  !TAB csv | caption | note        (csv: first row = header)
  !EQ latex | number
  [n] ...             reference entry (DOI turned into a hyperlink)
  *text*              italic, ^text^ superscript, ~text~ subscript
"""
import copy, re, sys
from pathlib import Path
import docx, latex2mathml.converter
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT.parent / "IGD11_Templat_Alanyauni_2026.docx"
SRC = ROOT / "manuscript/manuscript.md"; OUT = ROOT / "manuscript/manuscript.docx"
XSL = etree.XSLT(etree.parse(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"))

doc = docx.Document(TEMPLATE)
body = doc.element.body
children = list(body)
for el in children[6:]:            # keep header table, title table; drop instructional body
    if el.tag != qn("w:sectPr"):
        body.remove(el)
sect = body[-1]


def new_par(style=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line=0):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.first_line_indent = Cm(first_line); p.alignment = align
    body.remove(p._p); sect.addprevious(p._p)
    return p


def add_runs(p, text, bold=False, size=None):
    """Very small inline markup: *italic*, ^sup^, ~sub~."""
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~)", text):
        if not tok:
            continue
        if tok.startswith("**") and len(tok) > 4:
            r = p.add_run(tok[2:-2]); r.bold = True; r.font.color.rgb = RGBColor(0, 0, 0)
            if size: r.font.size = Pt(size)
            continue
        r = p.add_run(tok[1:-1] if tok[0] in "*^~" and len(tok) > 1 else tok)
        r.bold = bold or None; r.font.color.rgb = RGBColor(0, 0, 0)
        if size: r.font.size = Pt(size)
        if tok[0] == "*": r.italic = True
        if tok[0] == "^": r.font.superscript = True
        if tok[0] == "~": r.font.subscript = True
    return p


def blank():
    new_par()


M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def omml(latex):
    em = chr(0x2003); latex = latex.replace(r"\qquad", r"\text{" + em * 2 + "}").replace(r"\quad", r"\text{" + em + "}")
    mml = latex2mathml.converter.convert(latex)
    root = XSL(etree.fromstring(mml.encode())).getroot()
    # MML2OMML leaves the summand outside an n-ary operator; move it into <m:e>
    for nary in root.iter(f"{{{M_NS}}}nary"):
        e = nary.find(f"{{{M_NS}}}e")
        if e is None or len(e):
            continue
        sib = nary.getnext()
        while sib is not None:
            txt = "".join(sib.itertext())
            if sib.tag == f"{{{M_NS}}}r" and any(ch in txt for ch in ",=" + chr(0x2003)):
                break
            nxt = sib.getnext(); e.append(sib); sib = nxt
    return root


def equation(latex, num):
    """Borderless 3-cell row: the equation alone in the middle cell is rendered in display mode."""
    t = doc.add_table(rows=1, cols=3); body.remove(t._tbl); sect.addprevious(t._tbl)
    t.autofit = False
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); t._tbl.tblPr.append(lay)
    for j, wcm in enumerate([1.5, 14.0, 1.5]):
        t.columns[j].width = Cm(wcm); t.cell(0, j).width = Cm(wcm)
    mid = t.cell(0, 1).paragraphs[0]; mid.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mid.paragraph_format.first_line_indent = Cm(0)
    para = OxmlElement("m:oMathPara"); para.append(parse_xml(etree.tostring(omml(latex)).decode()))
    mid._p.append(para)
    right = t.cell(0, 2).paragraphs[0]; right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    right.paragraph_format.first_line_indent = Cm(0); add_runs(right, f"({num})")
    t.cell(0, 2).vertical_alignment = 1


def caption(kind, text, align):
    p = new_par(align=align)
    m = re.match(rf"({kind} \d+\.)\s*(.*)", text)
    add_runs(p, m.group(1), bold=True); add_runs(p, " " + m.group(2))
    p.paragraph_format.keep_together = True
    return p


def set_cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr(); b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders"); tcPr.append(b)
    for edge, val in kw.items():
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), val); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000")
        b.append(e)


WIDTHS = {"table1_samples.csv": [3.6, 3.3, 3.2, 3.4, 3.4],
          "table2_macro_f1.csv": [1.45, 2.85, 2.54, 2.54, 2.54, 2.54, 2.54],
          "table3_paired_delta.csv": [1.8, 3.8, 3.8, 3.8, 3.8]}


def table(csv, cap, note):
    blank(); caption("Table", cap, WD_ALIGN_PARAGRAPH.LEFT).paragraph_format.keep_with_next = True
    df = pd.read_csv(ROOT / csv, dtype=str, keep_default_na=False)
    t = doc.add_table(rows=len(df) + 1, cols=len(df.columns)); body.remove(t._tbl); sect.addprevious(t._tbl)
    t.alignment = 1
    for j, col in enumerate(df.columns):
        c = t.cell(0, j); c.text = ""; add_runs(c.paragraphs[0], col, bold=True)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ""; add_runs(c.paragraphs[0], str(v))
    for i, r in enumerate(t.rows):
        for j, c in enumerate(r.cells):
            p = c.paragraphs[0]; p.paragraph_format.first_line_indent = Cm(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1); p.paragraph_format.space_after = Pt(1)
            kw = {}
            if i == 0: kw.update(top="single", bottom="single")
            if i == len(t.rows) - 1: kw.update(bottom="single")
            if kw: set_cell_border(c, **kw)
            if i < len(t.rows) - 1:
                p.paragraph_format.keep_with_next = True   # keep the table on one page
    widths = WIDTHS.get(Path(csv).name)
    if widths:
        t.autofit = False
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); t._tbl.tblPr.append(lay)
        mar = OxmlElement("w:tblCellMar")
        for side in ("left", "right"):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), "57"); e.set(qn("w:type"), "dxa"); mar.append(e)
        t._tbl.tblPr.append(mar)
        for j, wcm in enumerate(widths):
            t.columns[j].width = Cm(wcm)
            for c in t.columns[j].cells:
                c.width = Cm(wcm)
    if note:
        p = new_par(align=WD_ALIGN_PARAGRAPH.LEFT); add_runs(p, note, size=9)
    blank()


def figure(path, width, cap):
    blank()
    p = new_par(align=WD_ALIGN_PARAGRAPH.CENTER)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(ROOT / path), width=Cm(float(width)))
    caption("Figure", cap, WD_ALIGN_PARAGRAPH.CENTER)
    blank()


def reference(text):
    p = new_par(align=WD_ALIGN_PARAGRAPH.LEFT)
    p.paragraph_format.left_indent = Cm(0.75); p.paragraph_format.first_line_indent = Cm(-0.75)
    m = re.search(r"(https://doi\.org/\S+)$", text)
    add_runs(p, text[: m.start()] if m else text)
    if m:
        url = m.group(1); rid = doc.part.relate_to(url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), rid)
        r = OxmlElement("w:r"); rpr = OxmlElement("w:rPr"); st = OxmlElement("w:rStyle"); st.set(qn("w:val"), "Hyperlink")
        rpr.append(st); r.append(rpr); t = OxmlElement("w:t"); t.text = url; r.append(t); h.append(r); p._p.append(h)


def set_cell_text(cell, text, bold=False, size=None, par=0):
    p = cell.paragraphs[par]
    for r in p.runs[1:]: r._r.getparent().remove(r._r)
    for h in p._p.findall(qn("w:hyperlink")): p._p.remove(h)
    if p.runs:
        p.runs[0].text = ""
        r0 = p.runs[0]
    add_runs(p, text, bold=bold, size=size)
    if p.runs and p.runs[0].text == "": p.runs[0]._r.getparent().remove(p.runs[0]._r)


# ---------------- citations ----------------
REFS = {}
for ln in (ROOT / "manuscript/references.txt").read_text(encoding="utf-8").splitlines():
    if "|" in ln:
        k, v = ln.split("|", 1); REFS[k.strip()] = v.strip()
ORDER = []


def fmt_nums(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(map(str, nums[i:j + 1])))
        i = j + 1
    return "[" + ", ".join(out) + "]"


def cite(text):
    def rep(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        for k in keys:
            if k not in REFS: sys.exit(f"unknown reference {k}")
            if k not in ORDER: ORDER.append(k)
        return fmt_nums([ORDER.index(k) + 1 for k in keys])
    return re.sub(r"\[(@[\w]+(?:;\s*@[\w]+)*)\]", rep, text)


# ---------------- parse source ----------------
lines = [cite(l) for l in SRC.read_text(encoding="utf-8").splitlines()]
front = {}; first_heading = True; decl = False
tt = doc.tables[1]
for ln in lines:
    s = ln.strip()
    if not s or s.startswith("<!--"):
        continue
    if s.startswith("@"):
        k, v = s[1:].split(" ", 1); front[k] = v; continue
    if s.startswith("### "):
        decl = True; blank(); p = new_par(align=WD_ALIGN_PARAGRAPH.LEFT); p.paragraph_format.keep_with_next = True
        add_runs(p, s[4:], bold=True); continue
    if s.startswith("## "):
        blank(); p = new_par(style="List Paragraph", align=WD_ALIGN_PARAGRAPH.LEFT)
        p.paragraph_format.left_indent = Cm(0); p.paragraph_format.keep_with_next = True
        add_runs(p, s[3:], bold=True); continue
    if s.startswith("# "):
        if not first_heading: blank()
        first_heading = False
        p = new_par(style="Heading 2", align=WD_ALIGN_PARAGRAPH.JUSTIFY); p.paragraph_format.space_before = Pt(0)
        add_runs(p, s[2:]); continue
    if s.startswith("!FIG"):
        a = [x.strip() for x in s[4:].split("|")]; figure(a[0], a[1], a[2]); continue
    if s.startswith("!TAB"):
        a = [x.strip() for x in s[4:].split("|")]; table(a[0], a[1], a[2] if len(a) > 2 else ""); continue
    if s.startswith("!EQ"):
        a = [x.strip() for x in s[3:].rsplit("|", 1)]; equation(a[0], a[1]); continue
    if s == "!REFS":
        unused = set(REFS) - set(ORDER)
        if unused: print("WARNING uncited references:", unused)
        for i, k in enumerate(ORDER, 1):
            reference(f"[{i}] {REFS[k]}")
        continue
    add_runs(new_par(align=WD_ALIGN_PARAGRAPH.LEFT if decl else WD_ALIGN_PARAGRAPH.JUSTIFY), s)

# ---------------- front matter ----------------
set_cell_text(tt.cell(0, 0), front["TITLE"], bold=True, size=14)
tt.cell(0, 0).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
set_cell_text(tt.cell(2, 1), front["CITE"], size=8)
set_cell_text(tt.cell(4, 1), front["ABSTRACT"])
# ---------------- author block (single author) ----------------
if "AUTHOR" in front:
    ac = tt.cell(1, 0); ps = ac.paragraphs


    def clear(p):
        for r in list(p.runs): r._r.getparent().remove(r._r)
        for h in p._p.findall(qn("w:hyperlink")): p._p.remove(h)


    def run(p, text, bold=None, italic=None, size=None, sup=False):
        r = p.add_run(text); r.bold = bold; r.italic = italic; r.font.superscript = sup or None
        if size: r.font.size = Pt(size)
        return r


    def link(p, url, size=8):
        rid = doc.part.relate_to(url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), rid)
        r = OxmlElement("w:r"); rpr = OxmlElement("w:rPr"); st = OxmlElement("w:rStyle"); st.set(qn("w:val"), "Hyperlink")
        sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(size * 2)); rpr.append(st); rpr.append(sz); r.append(rpr)
        t = OxmlElement("w:t"); t.text = url; r.append(t); h.append(r); p._p.append(h)


    clear(ps[0]); run(ps[0], "*", bold=True, sup=True); run(ps[0], "1", sup=True); run(ps[0], " " + front["AUTHOR"], bold=True)
    clear(ps[1]); run(ps[1], "1", italic=True, size=8, sup=True)
    run(ps[1], " " + front["AFFIL"] + ", e-mail: " + front["EMAIL"], italic=True, size=8)
    clear(ps[4]); run(ps[4], "1 ", size=8, sup=True); run(ps[4], "ORCID: ", size=8); link(ps[4], "https://orcid.org/" + front["ORCID"])
    clear(ps[6]); run(ps[6], "* Correspondence: ", size=8); run(ps[6], front["EMAIL"], size=8)
    for k in (5, 3, 2):
        ps[k]._p.getparent().remove(ps[k]._p)

kw = [k.strip() for k in front["KEYWORDS"].split(";")]
kc = tt.cell(5, 0)
for i, p in enumerate(kc.paragraphs[1:]):
    if i < len(kw):
        for r in p.runs[1:]: r._r.getparent().remove(r._r)
        p.runs[0].text = kw[i] + (";" if i < len(kw) - 1 else ".")
    else:
        p._p.getparent().remove(p._p)
doc.save(OUT)
print("saved", OUT)
