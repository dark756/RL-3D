"""Build docs/Mars_Lander_RL_code_guide.pdf: the call tree from main() and every node in in-order (LVR).

It checks every node against the source and stops if the code differs, so it only builds from the tree it
describes (commit 006e351); see README.md here.

    python tools/code_guide/make_code_guide.py --project <folder with that tree>
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from pathlib import Path

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--project", type=Path, default=SCR.parents[1],
                 help="project folder to document and write docs/ into (default: this repo)")
PROJECT = _ap.parse_args().project.resolve()
sys.path.insert(0, str(PROJECT))

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_RIGHT  # noqa: E402
from reportlab.lib.fonts import addMapping  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, KeepTogether, NextPageTemplate,  # noqa: E402
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

import guide_tree as gt  # noqa: E402
from lander.config import GROUPS, load_config  # noqa: E402

OUT = PROJECT / "docs" / "Mars_Lander_RL_code_guide.pdf"

# ------------------------------------------------------------------------------------------------ fonts
FONTS = r"C:\Windows\Fonts"
for name, file in (("Body", "segoeui.ttf"), ("Body-Bold", "segoeuib.ttf"), ("Body-Italic", "segoeuii.ttf"),
                   ("Body-BoldItalic", "segoeuiz.ttf"), ("Semi", "seguisb.ttf"), ("Sym", "seguisym.ttf"),
                   ("Mono", "consola.ttf"), ("Mono-Bold", "consolab.ttf"), ("Mono-Italic", "consolai.ttf")):
    pdfmetrics.registerFont(TTFont(name, f"{FONTS}\\{file}"))
addMapping("Body", 0, 0, "Body")
addMapping("Body", 1, 0, "Body-Bold")
addMapping("Body", 0, 1, "Body-Italic")
addMapping("Body", 1, 1, "Body-BoldItalic")
addMapping("Mono", 0, 0, "Mono")
addMapping("Mono", 1, 0, "Mono-Bold")
addMapping("Mono", 0, 1, "Mono-Italic")
addMapping("Mono", 1, 1, "Mono-Bold")

NAVY = colors.HexColor("#1F3B5C")
RUST = colors.HexColor("#A8432A")
INK = colors.HexColor("#1E1E1E")
GREY = colors.HexColor("#666666")
LIGHT = colors.HexColor("#F2F4F7")
RULE = colors.HexColor("#C9D1DB")
TEAL = colors.HexColor("#2E7D7A")
PURPLE = colors.HexColor("#5E4B8B")
BLUEFILL = colors.HexColor("#E3ECF6")
PURPLEFILL = colors.HexColor("#ECE7F5")
LINKC = "#1F5F99"

BASE = 9.0
ST = {
    "title": ParagraphStyle("title", fontName="Semi", fontSize=27, leading=33, textColor=NAVY),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=13, leading=18, textColor=GREY),
    "meta": ParagraphStyle("meta", fontName="Body", fontSize=10, leading=15, textColor=INK),
    "h1": ParagraphStyle("h1", fontName="Semi", fontSize=16, leading=20, spaceBefore=4, spaceAfter=8, textColor=NAVY),
    "h2": ParagraphStyle("h2", fontName="Semi", fontSize=11.5, leading=15, spaceBefore=9, spaceAfter=4,
                         textColor=NAVY),
    "body": ParagraphStyle("body", fontName="Body", fontSize=BASE, leading=BASE * 1.42, spaceAfter=5, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=BASE, leading=BASE * 1.38, leftIndent=12,
                             bulletIndent=2, spaceAfter=2.5, textColor=INK),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8, leading=10.3, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Semi", fontSize=8, leading=10.3, textColor=NAVY),
    "cellmono": ParagraphStyle("cellmono", fontName="Mono", fontSize=7.6, leading=10, textColor=INK),
    "caption": ParagraphStyle("caption", fontName="Body-Italic", fontSize=8, leading=10.5, textColor=GREY,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=8),
    "tree": ParagraphStyle("tree", fontName="Mono", fontSize=7.3, leading=9.0, textColor=INK),
    "treefile": ParagraphStyle("treefile", fontName="Body", fontSize=6.8, leading=9.0, textColor=GREY,
                               alignment=TA_RIGHT),
    "ehead": ParagraphStyle("ehead", fontName="Semi", fontSize=9.6, leading=12.5, textColor=NAVY),
    "ekind": ParagraphStyle("ekind", fontName="Body", fontSize=7.4, leading=12.5, textColor=GREY, alignment=TA_RIGHT),
    "enav": ParagraphStyle("enav", fontName="Body", fontSize=7.6, leading=10, textColor=GREY, spaceBefore=2,
                           spaceAfter=2),
    "esum": ParagraphStyle("esum", fontName="Body", fontSize=8.6, leading=11.6, textColor=INK, spaceAfter=3),
    "ecompact": ParagraphStyle("ecompact", fontName="Body", fontSize=8.2, leading=11, textColor=INK, spaceAfter=5,
                               leftIndent=8),
}
TOC_STYLES = [ParagraphStyle("toc0", fontName="Semi", fontSize=10, leading=15, leftIndent=0, textColor=NAVY),
              ParagraphStyle("toc1", fontName="Body", fontSize=9, leading=12.5, leftIndent=14, textColor=INK)]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(s: str) -> str:
    """Escape, then: `code` → mono, **x** → bold, [[REF]] → ↺, _{x} → subscript, ^{x} → superscript."""
    s = esc(s or "")
    s = re.sub(r"`([^`]+)`", r'<font name="Mono" size="7.8">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = s.replace("[[REF]]", SYM_REF)
    s = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^}]*)\}", r"<super>\1</super>", s)
    return s


SYM_REF = '<font name="Sym">\u21BA</font>'

# ------------------------------------------------------------------------------------------------ source index
INDEX: dict[str, dict] = {}


def _sig(fn: ast.FunctionDef) -> str:
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return ", ".join(names)


def index_file(path: Path) -> None:
    rel = path.relative_to(PROJECT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def visit(body, prefix, owner):
        for n in body:
            if isinstance(n, ast.ClassDef):
                INDEX[f"{rel}::{prefix}{n.name}"] = dict(node=n, line=n.lineno, kind="class", file=rel,
                                                         doc=ast.get_docstring(n) or "")
                visit(n.body, prefix + n.name + ".", "class")
            elif isinstance(n, ast.FunctionDef):
                prop = any(isinstance(d, ast.Name) and d.id == "property" for d in n.decorator_list)
                kind = "property" if prop else {"class": "method", "function": "closure"}.get(owner, "function")
                INDEX[f"{rel}::{prefix}{n.name}"] = dict(node=n, line=n.lineno, kind=kind, file=rel, sig=_sig(n),
                                                         doc=ast.get_docstring(n) or "")
                visit(n.body, prefix + n.name + ".", "function")

    visit(tree.body, "", None)


for folder in ("lander", "scripts", "tests"):
    for p in sorted((PROJECT / folder).glob("*.py")):
        index_file(p)


def called_names(fn_node) -> set[str]:
    out = set()
    for x in ast.walk(fn_node):
        if isinstance(x, ast.Attribute):
            out.add(x.attr)
        elif isinstance(x, ast.Name):
            out.add(x.id)
    return out


def policy_line() -> int:
    cls = INDEX["lander/scenarios.py::Scenario"]["node"]
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "policy":
            return n.lineno
    return cls.lineno


# ------------------------------------------------------------------------------------------------ tree prep
LAMBDA_KEY = "lambda::Scenario.policy"
HOMES: dict[str, gt.Node] = {}
ALL: list[gt.Node] = []


def prepare(n: gt.Node, parent=None) -> None:
    n.parent = parent
    if n.kind == "lambda":
        n.key, n.call_name, n.verify = LAMBDA_KEY, "policy", True
    if n.kind == "ref-lambda":
        n.kind, n.key = "ref", LAMBDA_KEY
    if n.kind == "source":
        info = INDEX.get(n.key)
        if info is None:
            raise SystemExit(f"not in source: {n.key}")
        n.src = info
        n.kind = info["kind"]
    if n.kind not in ("ref", "external") and n.key:
        if n.key in HOMES:
            raise SystemExit(f"two homes for {n.key}")
        HOMES[n.key] = n
    ALL.append(n)
    for c in n.children:
        prepare(c, n)


prepare(gt.ROOT)
for n in ALL:
    if n.kind == "ref":
        if n.key not in HOMES:
            raise SystemExit(f"reference without a home: {n.key}")
        n.target = HOMES[n.key]
        if n.ins is None:
            n.ins = n.target.ins
        if n.outs is None:
            n.outs = n.target.outs

SOURCE_KINDS = ("function", "method", "property", "closure")


def qual(n: gt.Node) -> str:
    return n.key.split("::", 1)[1]


def label(n: gt.Node) -> str:
    if n.kind == "ref":
        return label(n.target)
    if n.label:
        return n.label
    if n.kind in SOURCE_KINDS:
        file, q = n.key.split("::", 1)
        if file.startswith("scripts/") and q == "main":
            return f"{Path(file).stem}.main"
        return q
    return n.key or "?"


def signature(n: gt.Node) -> str:
    base = label(n)
    if n.kind == "property":
        return base
    if n.kind in SOURCE_KINDS:
        return f"{base}({n.src['sig']})"
    return base


def where(n: gt.Node) -> str:
    t = n.target if n.kind == "ref" else n
    if t.kind in SOURCE_KINDS:
        return f"{t.src['file']}:{t.src['line']}"
    if t.key == LAMBDA_KEY:
        return f"lander/scenarios.py:{policy_line()}"
    if t.kind == "testfile":
        return t.label
    return ""


def simple_name(n: gt.Node) -> str:
    if n.kind == "external" or n.key == LAMBDA_KEY:
        return n.call_name or "policy"
    t = n.target if n.kind == "ref" else n
    if t.key == LAMBDA_KEY:
        return "policy"
    q = qual(t).split(".")
    if q[-1] in ("__init__", "__post_init__"):
        return q[-2]
    return q[-1]


# verify that every edge is a real call in the parent's source
problems = []
for n in ALL:
    p = n.parent
    if p is None or not n.verify or p.kind not in SOURCE_KINDS:
        continue
    if simple_name(n) not in called_names(p.src["node"]):
        problems.append(f"{label(p)} -> {label(n)} ('{simple_name(n)}')")
if problems:
    raise SystemExit("edges not found in source:\n  " + "\n  ".join(problems))

# in-order (LVR) numbering for n-ary nodes: first child's subtree, the node, then the remaining subtrees
ORDER: list[gt.Node] = []


def inorder(n: gt.Node) -> None:
    if n.children:
        inorder(n.children[0])
        ORDER.append(n)
        for c in n.children[1:]:
            inorder(c)
    else:
        ORDER.append(n)


inorder(gt.ROOT)
for i, n in enumerate(ORDER, 1):
    n.num = i

# functions defined in lander/ and scripts/ that are on no call path from main
on_tree = {n.key for n in ALL if n.kind in SOURCE_KINDS}
UNUSED = []
for key, info in INDEX.items():
    if not (key.startswith("lander/") or key.startswith("scripts/")) or info["kind"] not in SOURCE_KINDS:
        continue
    name = key.split("::")[1].split(".")[-1]
    if name.startswith("__") and name not in ("__init__", "__post_init__"):
        continue
    if key not in on_tree:
        UNUSED.append(key)


def link(n: gt.Node, with_name=False) -> str:
    t = f'<a href="#n{n.num}" color="{LINKC}">[{n.num}]</a>'
    return f"{t} {esc(label(n))}" if with_name else t


# ------------------------------------------------------------------------------------------------ drawings
def _box(d, x, y, w, h, text, fill, stroke, color, font="Body", size=7.2, bold=False):
    d.add(Rect(x, y, w, h, rx=3, ry=3, fillColor=fill, strokeColor=stroke, strokeWidth=0.6))
    d.add(String(x + 5, y + h / 2 - size * 0.35, text, fontName="Semi" if bold else font, fontSize=size,
                 fillColor=color))


def _arrow(d, x1, y1, x2, y2, color=GREY, width=0.8):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    s = 4.2
    p1 = (x2 - s * math.cos(ang - 0.42), y2 - s * math.sin(ang - 0.42))
    p2 = (x2 - s * math.cos(ang + 0.42), y2 - s * math.sin(ang + 0.42))
    d.add(Polygon([x2, y2, p1[0], p1[1], p2[0], p2[1]], fillColor=color, strokeColor=color, strokeWidth=0.3))


def node_style(n: gt.Node):
    """fill, stroke, text colour for diagram boxes."""
    if n.kind == "entry":
        return NAVY, NAVY, colors.white
    if n.kind in ("tests", "testfile", "notebook"):
        return PURPLEFILL, PURPLE, PURPLE
    if n.kind == "ref":
        return colors.white, TEAL, TEAL
    if n.kind == "external":
        return colors.white, RULE, GREY
    if n.parent is gt.ROOT:
        return BLUEFILL, NAVY, NAVY
    return colors.white, RULE, INK


def short(n: gt.Node) -> str:
    if n.kind == "ref":
        return f"[{n.num}] \u21BA {label(n)}  \u2192 [{n.target.num}]"
    return f"[{n.num}] {label(n)}"


def overview_drawing(width: float) -> Drawing:
    root = gt.ROOT
    row_h = 12.6
    pos = {}
    y = 0.0
    for c in root.children:
        if c.children:
            ys = []
            for g in c.children:
                pos[id(g)] = (2, y)
                ys.append(y)
                y += 1
            pos[id(c)] = (1, sum(ys) / len(ys))
        else:
            pos[id(c)] = (1, y)
            y += 1
        y += 0.45
    ys1 = [pos[id(c)][1] for c in root.children]
    pos[id(root)] = (0, (min(ys1) + max(ys1)) / 2)
    height = y * row_h + 8
    d = Drawing(width, height)
    xs, ws = {0: 0.0, 1: 66.0, 2: 252.0}, {0: 44.0, 1: 162.0, 2: width - 252.0}
    bh = 10.2

    def yc(yi):
        return height - 4 - yi * row_h - row_h / 2

    def draw(n):
        lvl, yi = pos[id(n)]
        fill, stroke, col = node_style(n)
        sym = n.kind == "ref"
        txt = short(n)
        x, w = xs[lvl], ws[lvl]
        d.add(Rect(x, yc(yi) - bh / 2, w, bh, rx=2.5, ry=2.5, fillColor=fill, strokeColor=stroke, strokeWidth=0.6))
        font = "Sym" if sym else ("Semi" if lvl < 2 else "Body")
        d.add(String(x + 4, yc(yi) - 2.4, txt, fontName=font, fontSize=6.9, fillColor=col))

    for c in root.children:
        lvl, yi = pos[id(c)]
        _, ry = pos[id(root)]
        xm = xs[0] + ws[0] + 9
        d.add(Line(xs[0] + ws[0], yc(ry), xm, yc(ry), strokeColor=GREY, strokeWidth=0.6))
        d.add(Line(xm, yc(ry), xm, yc(yi), strokeColor=GREY, strokeWidth=0.6))
        d.add(Line(xm, yc(yi), xs[1], yc(yi), strokeColor=GREY, strokeWidth=0.6))
        for g in c.children:
            _, gy = pos[id(g)]
            xm2 = xs[1] + ws[1] + 10
            d.add(Line(xs[1] + ws[1], yc(yi), xm2, yc(yi), strokeColor=GREY, strokeWidth=0.6))
            d.add(Line(xm2, yc(yi), xm2, yc(gy), strokeColor=GREY, strokeWidth=0.6))
            d.add(Line(xm2, yc(gy), xs[2], yc(gy), strokeColor=GREY, strokeWidth=0.6))
    draw(root)
    for c in root.children:
        draw(c)
        for g in c.children:
            draw(g)
    return d


def pipeline_drawing(width: float) -> Drawing:
    n_facts = len(load_config(environ={}).params)
    items = [
        (".env", [f"{n_facts} facts", "value | unit | status", "| source | note"]),
        ("load_config()", ["parses and validates", "→ Config", f"({len(GROUPS)} typed groups)"]),
        ("LanderSim(cfg)", ["build_geometry", "build_model → MuJoCo", "vehicle layer objects"]),
        ("reset / step", ["reset(seed, state,", "options), then", "step(Command) × N"]),
        ("StepResult", ["telemetry, sensors,", "touchdown report,", "events"]),
        ("consumers", ["viewer · MP4 video", "summary · tests", "notebook"]),
    ]
    n = len(items)
    gap = 12
    w = (width - gap * (n - 1)) / n
    h = 52
    d = Drawing(width, h + 6)
    for i, (head, lines) in enumerate(items):
        x = i * (w + gap)
        fill = BLUEFILL if i in (2, 3) else LIGHT
        d.add(Rect(x, 3, w, h, rx=4, ry=4, fillColor=fill, strokeColor=NAVY if i in (2, 3) else RULE,
                   strokeWidth=0.7))
        d.add(String(x + w / 2, h - 10, head, fontName="Semi", fontSize=8, fillColor=NAVY, textAnchor="middle"))
        for j, t in enumerate(lines):
            d.add(String(x + w / 2, h - 22 - j * 9.5, t, fontName="Body", fontSize=6.7, fillColor=INK,
                         textAnchor="middle"))
        if i < n - 1:
            _arrow(d, x + w + 1, 3 + h / 2, x + w + gap - 1, 3 + h / 2, NAVY)
    return d


def step_flow_drawing(width: float) -> Drawing:
    rows = [
        ("LanderSim.step(Command)", "Vehicle.command: valve settings → thrust targets; engines_off → shut down for good",
         False),
        ("1  read the state", "qpos, qvel → rotation R, CG = origin + R·cg (fixed), CG velocity", True),
        ("2  Vehicle.step", "dt → actual thrust (3) after the 53 ms lag, force_b, torque_b about the CG, ṁ, empty", True),
        ("3  drag", "CG velocity − steady wind → v_rel (body axes) → drag_force_body → F_aero, acting at the CG", True),
        ("4  write forces", "xfrc_applied ← R·(force_b + F_aero) + disturbance, R·torque_b; body mass ← dry + "
                            "propellant", True),
        ("5  mujoco.mj_step", "dynamics + gravity + soft contacts → new qpos, qvel, contacts, sensors", True),
        ("6  contacts, events, monitor", "_contacts → pad / body contact; _flag_events; note_engines; "
                                         "TouchdownMonitor.update → report", True),
        ("_result", "mj_forward, then telemetry() + read_sensors() + report + events → StepResult", False),
    ]
    row_h, gap, pad = 22, 6, 8
    height = len(rows) * (row_h + gap) + 2 * pad + 16
    d = Drawing(width, height)
    ys, y = [], height - 2 - row_h
    for i, (_, _, inner) in enumerate(rows):
        if i == 1:
            y -= pad + 10
        if i == len(rows) - 1:
            y -= pad
        ys.append(y)
        y -= row_h + gap
    top, bottom = ys[1] + row_h + pad + 10, ys[-2] - pad
    d.add(Rect(4, bottom, width - 8, top - bottom, rx=6, ry=6, fillColor=colors.HexColor("#F7F9FC"),
               strokeColor=NAVY, strokeWidth=0.6, strokeDashArray=[3, 2]))
    d.add(String(16, top - 9, "× 20 physics steps of 2 ms (LanderSim._physics_step)", fontName="Semi",
                 fontSize=7.2, fillColor=NAVY, textAnchor="start"))
    for i, (head, text, inner) in enumerate(rows):
        x = 16 if inner else 0
        w = width - (32 if inner else 0)
        fill = colors.white if inner else BLUEFILL
        d.add(Rect(x, ys[i], w, row_h, rx=3, ry=3, fillColor=fill, strokeColor=RULE if inner else NAVY,
                   strokeWidth=0.6))
        d.add(String(x + 6, ys[i] + row_h - 9.3, head, fontName="Semi", fontSize=7.6, fillColor=NAVY))
        d.add(String(x + 6, ys[i] + 4.3, text, fontName="Body", fontSize=6.8, fillColor=INK))
        if i < len(rows) - 1:
            ax = width - 30
            _arrow(d, ax, ys[i] - 0.5, ax, ys[i + 1] + row_h + 0.5, NAVY, 0.7)
    return d


def lvr_drawing(width: float) -> Drawing:
    d = Drawing(width, 112)
    nodes = {"P": (width / 2, 92, "P  [3]"), "A": (width / 2 - 110, 56, "A  [2]"), "B": (width / 2, 56, "B  [4]"),
             "C": (width / 2 + 110, 56, "C  [6]"), "a1": (width / 2 - 110, 18, "a1  [1]"),
             "c1": (width / 2 + 70, 18, "c1  [5]"), "c2": (width / 2 + 150, 18, "c2  [7]")}
    for a, b in (("P", "A"), ("P", "B"), ("P", "C"), ("A", "a1"), ("C", "c1"), ("C", "c2")):
        d.add(Line(nodes[a][0], nodes[a][1] - 7, nodes[b][0], nodes[b][1] + 7, strokeColor=GREY, strokeWidth=0.7))
    for key, (x, y, t) in nodes.items():
        fill = BLUEFILL if key == "P" else colors.white
        d.add(Rect(x - 26, y - 7, 52, 14, rx=3, ry=3, fillColor=fill, strokeColor=NAVY, strokeWidth=0.6))
        d.add(String(x, y - 2.6, t, fontName="Semi", fontSize=7.4, fillColor=NAVY, textAnchor="middle"))
    d.add(String(width / 2 - 110 - 30, 92 - 2.6, "L = first child's subtree", fontName="Body", fontSize=7,
                 fillColor=GREY, textAnchor="end"))
    d.add(String(width / 2 + 110 + 30, 92 - 2.6, "R = the other children's subtrees", fontName="Body", fontSize=7,
                 fillColor=GREY))
    return d


# ------------------------------------------------------------------------------------------------ document
class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=19 * mm,
                         bottomMargin=17 * mm, title="Mars Lander RL - code guide (M1)",
                         author="Mars Lander RL project",
                         subject="Call tree from main() with every node explained in in-order (LVR)")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f", leftPadding=0,
                      rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="cover", frames=[frame]),
                               PageTemplate(id="body", frames=[frame], onPage=self._decor)])

    def _decor(self, canv, doc):
        canv.saveState()
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.5)
        y = A4[1] - 12 * mm
        canv.line(self.leftMargin, y, A4[0] - self.rightMargin, y)
        canv.setFont("Body", 7.5)
        canv.setFillColor(GREY)
        canv.drawString(self.leftMargin, y + 3, "Mars Lander RL  ·  code guide  ·  milestone M1, simplified")
        canv.drawRightString(A4[0] - self.rightMargin, y + 3, "call tree from main(), in-order (LVR)")
        canv.drawCentredString(A4[0] / 2, 9 * mm, str(doc.page))
        canv.restoreState()

    def afterFlowable(self, f):
        if hasattr(f, "_toc"):
            level, text, key = f._toc
            self.notify("TOCEntry", (level, text, self.page, key))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
        elif hasattr(f, "_outline"):
            level, text, key = f._outline
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=True)


def H1(text, key):
    p = Paragraph(f'<a name="{key}"/>{fmt(text)}', ST["h1"])
    p._toc = (0, text, key)
    return p


def H2(text, key):
    p = Paragraph(f'<a name="{key}"/>{fmt(text)}', ST["h2"])
    p._toc = (1, text, key)
    return p


def para(text, style="body"):
    return Paragraph(fmt(text), ST[style])


def bullets(items, style="bullet"):
    return [Paragraph(fmt(t), ST[style], bulletText="•") for t in items]


def table(rows, widths, header=True, mono_first=False):
    data = []
    for i, r in enumerate(rows):
        cells = []
        for j, c in enumerate(r):
            if header and i == 0:
                cells.append(Paragraph(fmt(c), ST["cellb"]))
            elif mono_first and j == 0:
                cells.append(Paragraph(fmt(c), ST["cellmono"]))
            else:
                cells.append(Paragraph(fmt(c), ST["cell"]))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY)]
    t.setStyle(TableStyle(style))
    return t


# ------------------------------------------------------------------------------------------------ content: data types
FIELD_TEXT = {
    "Param": {
        "key": "the .env key, e.g. VIKING_ENGINE_THRUST_MAX_N",
        "value": "int, float or tuple of floats",
        "unit": "unit text from the .env comment",
        "status": "published | derived | assumed (| override)",
        "source": "where to check it: S-code and page / figure / table",
        "note": "free text; for an override: 'was <old value> (<status>; <source>)'",
        "origin": "'.env', 'env:LANDER_<KEY>', 'cli', 'code' or 'test'",
    },
    "InitialState": {
        "position": "CG position, world frame (m)",
        "velocity": "CG velocity, world frame (m/s)",
        "quat": "attitude body → world (w, x, y, z)",
        "omega": "angular velocity in body axes (rad/s)",
        "propellant_kg": "propellant at the start; None = VIKING_MASS_START_PROPELLANT_KG (60 kg)",
        "throttle": "starting valve settings (3); None = hover trim",
    },
    "ResetOptions": {
        "wind_speed_mps": "steady wind speed; 0 = calm (dust storms are 17–30 m/s)",
        "wind_direction_deg": "direction the wind blows toward",
        "slope_deg": "ground slope",
        "slope_direction_deg": "downhill direction",
        "friction": "footpad friction coefficient",
        "disturbance_force_n": "constant extra force, world frame (N)",
        "disturbance_torque_nm": "constant extra torque, body frame (N·m)",
        "engine_thrust_scale": "per-engine fault factor (1 nominal, 0 failed)",
        "engines_enabled": "False = start with the engines already shut down (free fall)",
        "sensor_noise_scale": "None = the .env value (1 = Viking's radar errors, 0 = perfect readings)",
    },
    "Command": {
        "throttle": "3 Viking valve settings 0 … 1 (0 = 276 N, 1 = 2,667 N per engine)",
        "engines_off": "True shuts all three engines down for good (no relight)",
    },
    "VehicleStep": {
        "thrust": "actual thrust per engine after the lag, clamp and faults (N)",
        "force_b": "total thrust force, body frame (N)",
        "torque_b": "thrust torque about the CG, body frame (N·m)",
        "mdot": "propellant flow this step (kg/s)",
        "empty": "the tank has run dry",
    },
    "Telemetry": {
        "t": "simulation time (s)", "cg": "CG position, world (m)", "vel": "CG velocity, world (m/s)",
        "quat": "attitude (w, x, y, z)", "omega_b": "body rates (rad/s)", "tilt_deg": "thrust axis from vertical",
        "height_cg": "CG height above the ground below it (m)",
        "pad_altitude": "lowest footpad above the ground (m; 0 at contact)", "propellant_kg": "propellant left",
        "mass_kg": "total mass = dry + propellant", "thrust_n": "actual thrust per engine (N)",
        "throttle_cmd": "last valve settings", "engines_off": "engines shut down (command or empty tank)",
        "wind": "the episode's steady wind, world (m/s)", "pad_contact": "footpad contact flags (3)",
        "body_contact": "any body part touching the ground",
    },
    "SensorReadings": {
        "t": "sample time (s)", "accel": "specific force at the CG, body frame (m/s²); ideal IMU",
        "gyro": "body rates (rad/s); ideal IMU",
        "altitude": "radar-altimeter range to the nearest ground point (m), NaN = invalid (e.g. below 41 m)",
        "doppler_beam": "4 Doppler beam closing speeds (m/s), NaN = no lock",
        "doppler_vel": "body-frame ground velocity from ≥ 3 locked beams (m/s), NaN otherwise",
    },
    "TouchdownReport": {
        "touched_down": "a footpad or the body has touched", "first_contact_time": "time of first contact (s)",
        "vz_at_contact": "descent speed at first contact (m/s, + down)",
        "vh_at_contact": "horizontal speed at first contact (m/s)", "tilt_at_contact_deg": "tilt at first contact",
        "leg_broken": "first contact faster than 3.34 m/s down or 1.22 m/s sideways",
        "min_stability_m": "minimum stability distance (m)",
        "settled": "resting on all three feet, kinetic energy < 0.5 % of impact energy for 0.5 s",
        "settle_time": "time settled (s)",
        "tipped_over": "stability distance went negative", "body_strike": "a body part hit the ground",
        "propellant_exhausted_airborne": "the tank ran dry before touchdown",
        "engines_off": "engines shut down (must be True when the episode ends)",
        "final_tilt_deg": "tilt when the landing was decided", "final_xy": "position when decided (m)",
        "criteria": "name → True / False / None (None = decided at M2, O3)",
    },
    "StepResult": {
        "telemetry": "Telemetry (ground truth)", "sensors": "SensorReadings",
        "report": "TouchdownReport (live object)",
        "events": "this step's events: touchdown, leg_broken, engines_cut, settled, tipped_over, "
                  "propellant_exhausted, body_strike",
    },
    "Scenario": {
        "name": "drop | hard_drop | slope45 | hover | freefall", "description": "one-line description",
        "duration_s": "maximum simulated time (s)", "initial_state": "InitialState",
        "options": "dict of ResetOptions fields",
        "policy": "sim → Command (hover_then_cut; hover for the hover scenario)", "seed": "random seed",
    },
    "LanderGeometry": {
        "footpads": "footpad contact-sphere centres (3, 3)",
        "footpad_radius": "contact-sphere radius = junction height (0.03175 m)",
        "leg_azimuth_rad": "leg directions (3)",
        "cg": "centre of gravity on the centre line (0, 0, 0.7904 m)",
        "engine_pos": "engine thrust points on the underside (3, 3)",
        "radar_pos": "altimeter and Doppler antennas: centre of the underside",
        "doppler_beams": "beam directions (4, 3), body frame",
    },
    "ModelIndex": {
        "lander_body": "body id", "pad_geoms": "footpad sphere geom ids",
        "body_geoms": "hull, radar housing, engine and tank geom ids",
        "terrain_geom": "ground plane geom id", "ground_mocap": "mocap index of the ground body",
        "engine_sites": "engine site ids",
        "imu_acc_adr": "sensordata address of the IMU accelerometer",
        "imu_gyro_adr": "sensordata address of the IMU gyro",
    },
}
TYPE_FILES = {"Param": "lander/config.py", "InitialState": "lander/sim.py", "ResetOptions": "lander/sim.py",
              "Command": "lander/vehicle.py", "VehicleStep": "lander/vehicle.py", "Telemetry": "lander/sim.py",
              "SensorReadings": "lander/sensors.py", "TouchdownReport": "lander/touchdown.py",
              "StepResult": "lander/sim.py", "Scenario": "lander/scenarios.py",
              "LanderGeometry": "lander/geometry.py", "ModelIndex": "lander/mjcf.py"}


def dataclass_fields(cls_key):
    cls = INDEX[cls_key]["node"]
    out = []
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.append((n.target.id, ast.unparse(n.annotation)))
    return out


def type_tables():
    flow = []
    for name, file in TYPE_FILES.items():
        key = f"{file}::{name}"
        fields_ = dataclass_fields(key)
        text = FIELD_TEXT[name]
        missing = [f for f, _ in fields_ if f not in text]
        if missing:
            raise SystemExit(f"describe fields {missing} of {name}")
        doc = INDEX[key]["doc"].split("\n")[0]
        rows = [["field", "type", "meaning"]] + [[f, t, text[f]] for f, t in fields_]
        props = [q.split(".")[-1] for q, info in INDEX.items()
                 if q.startswith(key + ".") and info["kind"] == "property"]
        flow.append(KeepTogether([
            Paragraph(f'<font name="Mono" size="9" color="#1F3B5C"><b>{name}</b></font>'
                      f'<font name="Body" size="7.5" color="#666666">   {file}:{INDEX[key]["line"]}</font>', ST["h2"]),
            para(doc + (f"  Properties: {', '.join(props)}." if props else "")),
            table(rows, [92, 112, 312], mono_first=True), Spacer(1, 4)]))
    return flow


# ------------------------------------------------------------------------------------------------ content: entries
KIND_LABEL = {"function": "function", "method": "method", "property": "property", "closure": "nested function",
              "lambda": "lambda", "entry": "program entry", "tests": "test suite", "testfile": "test file",
              "notebook": "notebook", "external": "external library call", "ref": "reference"}


def nav_line(n):
    parts = []
    if n.parent is not None:
        parts.append(f"Parent {link(n.parent, True)}")
    if n.children:
        parts.append(f"L {link(n.children[0], True)}")
        if len(n.children) > 1:
            parts.append("R " + ", ".join(link(c, True) for c in n.children[1:]))
    return "  ·  ".join(parts)


def calls_cell(n):
    lines = []
    for c in n.children:
        ins = c.ins if c.ins is not None else ""
        outs = c.outs if c.outs is not None else ""
        name = esc(label(c))
        if c.kind == "ref":
            name = f"{SYM_REF} {name}"
        lines.append(f'→ {link(c)} <b>{name}</b> — gets: {fmt(ins)}; returns: {fmt(outs)}')
    return "<br/>".join(lines)


def full_entry(n):
    kind = KIND_LABEL.get(n.kind, n.kind)
    loc = where(n)
    head = Table([[Paragraph(f'<a name="n{n.num}"/><font color="#A8432A">[{n.num}]</font>  '
                             f'{esc(signature(n))}', ST["ehead"]),
                   Paragraph(f"{kind}{'  ·  ' + loc if loc and n.kind != 'testfile' else ''}", ST["ekind"])]],
                 colWidths=[372, 144])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                              ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                              ("LINEABOVE", (0, 0), (-1, 0), 0.8, NAVY)]))
    head._outline = (1, f"[{n.num}] {label(n)}", f"n{n.num}")
    items = [head, Paragraph(nav_line(n), ST["enav"]), Paragraph(fmt(n.summary), ST["esum"])]
    if n.kind == "testfile":
        tests = sorted((info["line"], q.split("::")[1]) for q, info in INDEX.items()
                       if q.startswith(n.label + "::test_") and info["kind"] == "function")
        if tests:
            items.append(Paragraph("<b>Tests:</b> " + ", ".join(f'<font name="Mono" size="7.6">{esc(t)}</font>'
                                                                for _, t in tests), ST["esum"]))
    rows = [["Receives", f"{fmt(n.ins)}" + (f'  <font color="#666666">(from {link(n.parent)})</font>'
                                             if n.parent is not None else "")],
            ["Returns", fmt(n.outs)]]
    if n.state:
        rows.append(["State", fmt(n.state)])
    if n.note:
        rows.append(["Note", fmt(n.note)])
    if n.children:
        rows.append(["Calls", calls_cell(n)])
    data = [[Paragraph(f"<b>{r[0]}</b>", ST["cellb"]), Paragraph(r[1], ST["cell"])] for r in rows]
    t = Table(data, colWidths=[56, 460])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)]))
    items += [t, Spacer(1, 7)]
    return KeepTogether(items)


def compact_entry(n):
    if n.kind == "ref":
        txt = (f'<a name="n{n.num}"/><font color="#A8432A"><b>[{n.num}]</b></font>  {SYM_REF} '
               f'<b>{esc(label(n))}</b>  <font color="#666666">reference: explained at</font> {link(n.target)}.  '
               f'Called by {link(n.parent, True)} — gets: {fmt(n.ins)}; returns: {fmt(n.outs)}.')
    else:
        txt = (f'<a name="n{n.num}"/><font color="#A8432A"><b>[{n.num}]</b></font>  '
               f'<font color="#666666">external</font>  <font name="Mono" size="7.8">{esc(n.label)}</font> — '
               f'{fmt(n.summary)} Called by {link(n.parent, True)} — gets: {fmt(n.ins)}; returns: {fmt(n.outs)}.')
    p = Paragraph(txt, ST["ecompact"])
    p._outline = (1, f"[{n.num}] {'ref ' if n.kind == 'ref' else 'ext '}{label(n)}", f"n{n.num}")
    return p


# ------------------------------------------------------------------------------------------------ content: tree outline
def outline_rows():
    rows = []

    def walk(n, prefix, last, root):
        conn = "" if root else ("\u2514\u2500 " if last else "\u251C\u2500 ")
        pre = (prefix + conn).replace(" ", "\u00A0")
        num = f'<a href="#n{n.num}" color="{LINKC}">[{n.num}]</a>'
        if n.kind == "ref":
            body = (f'<font color="#2E7D7A">{SYM_REF} {esc(label(n))} \u2192 </font>'
                    f'<a href="#n{n.target.num}" color="#2E7D7A">[{n.target.num}]</a>')
        elif n.kind == "external":
            body = f'<font color="#7A7A7A"><i>{esc(n.label)}</i></font>'
        elif n.kind in ("entry",) or n.parent is gt.ROOT:
            body = f'<b><font color="#1F3B5C">{esc(label(n))}</font></b>'
        elif n.kind in ("tests", "testfile", "notebook"):
            body = f'<font color="#5E4B8B">{esc(label(n))}</font>'
        else:
            body = esc(label(n))
        rows.append([Paragraph(f'<font color="#9AA5B1">{pre}</font>{num} {body}', ST["tree"]),
                     Paragraph(esc(where(n)) if n.kind != "ref" else "", ST["treefile"])])
        child_prefix = prefix + ("" if root else ("   " if last else "\u2502  "))
        for i, c in enumerate(n.children):
            walk(c, child_prefix, i == len(n.children) - 1, False)

    walk(gt.ROOT, "", True, True)
    t = Table(rows, colWidths=[402, 114], repeatRows=0)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 0.2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0.2), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#FAFBFC")])]))
    return t


# ------------------------------------------------------------------------------------------------ build
def check_glyphs():
    """Every character used in node texts and field descriptions must exist in the body font."""
    cmap = pdfmetrics.getFont("Body").face.charToGlyph
    texts = []
    for n in ALL:
        texts += [n.summary or "", n.ins or "", n.outs or "", n.state or "", n.note or "", n.label or ""]
    for d in FIELD_TEXT.values():
        texts += list(d.values())
    missing = sorted({c for t in texts for c in t if ord(c) > 127 and ord(c) not in cmap})
    if missing:
        raise SystemExit("characters missing from Segoe UI: " + " ".join(f"U+{ord(c):04X}" for c in missing))


def build():
    cfg = load_config(environ={})
    statuses = Counter(p.status for p in cfg.params.values())
    n_nodes = len(ORDER)
    n_src = sum(1 for n in ALL if n.kind in SOURCE_KINDS)
    n_ref = sum(1 for n in ALL if n.kind == "ref")
    n_ext = sum(1 for n in ALL if n.kind == "external")
    W = A4[0] - 34 * mm

    story = []
    # ---------------------------------------------------------------- cover
    story += [Spacer(1, 52 * mm),
              Paragraph("Mars Lander RL", ST["title"]),
              Spacer(1, 3),
              Paragraph("How the code fits together", ST["title"]),
              Spacer(1, 10),
              Paragraph("The call tree from <b>main()</b>, with every node explained in in-order (LVR) together "
                        "with the data it receives from its caller and passes to the functions it calls.",
                        ST["subtitle"]),
              Spacer(1, 26 * mm)]
    meta = [["Project", "Viking lander (1976) terminal descent on Mars · MuJoCo · Stable-Baselines3"],
            ["Milestone", "M1 — simulator, viewer, video, scripted scenarios, 37 tests (no RL yet); simplified on "
                          "5 October 2026 (rigid legs, fixed CG, no roll jets, the agent cuts the engines)"],
            ["Code", "lander/ (13 modules) · scripts/ (6) · tests/ (7 files + conftest) · 1 notebook"],
            ["Tree", f"{n_nodes} nodes: {n_src} functions explained once, {n_ref} reference nodes, "
                     f"{n_ext} external library calls, plus the entry, test and notebook nodes"],
            ["Facts", f"{len(cfg.params)} values in .env — {statuses['published']} published, "
                      f"{statuses['derived']} derived, {statuses['assumed']} assumed"],
            ["Versions", "Python 3.13.5 · MuJoCo 3.14.0 · numpy 2.5.3"],
            ["Date", "6 October 2026"]]
    mt = Table([[Paragraph(f"<b>{a}</b>", ST["cellb"]), Paragraph(fmt(b), ST["cell"])] for a, b in meta],
               colWidths=[70, 380])
    mt.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story += [mt, NextPageTemplate("body"), PageBreak()]

    # ---------------------------------------------------------------- contents
    toc = TableOfContents()
    toc.levelStyles = TOC_STYLES
    story += [Paragraph("Contents", ST["h1"]), toc, PageBreak()]

    # ---------------------------------------------------------------- 1 overview
    story += [H1("1  The project in one page", "s1")]
    story += [para("Mars Lander RL simulates NASA's Viking lander (1976) in the final few hundred metres of its "
                   "landing on Mars, so that a reinforcement-learning agent can later learn to land it in calm air and "
                   "in dust storms, and a fuzzer can search for the conditions where it fails. Milestone M1, the code "
                   "this guide describes, is the simulator: physics, a simplified Viking vehicle (rigid legs that "
                   "break above 3.34 m/s down or 1.22 m/s sideways, the CG fixed on the centre line, no roll jets, "
                   "engines that the pilot shuts off), Mars air and steady wind, radar sensors with Viking's errors, "
                   "a live 3-D viewer, MP4 export, scripted test scenarios and the test suite. There is no manual "
                   "flight: commands always come from code (scripted policies now, the agent from M4)."),
              para("Data moves in one direction (Figure 1). Every number lives in `.env` with its unit, status and "
                   "source. `load_config` turns the file into a typed `Config`. `LanderSim` builds the geometry, "
                   "generates the MuJoCo model from it and creates the backend-agnostic vehicle layer. An episode is "
                   "`reset(seed, initial state, options)` followed by `step(Command)` every 40 ms; each step runs 20 "
                   "physics steps of 2 ms (Figure 2) and returns a `StepResult`. The scripts, tests and notebook only "
                   "consume StepResults.")]
    story += [Spacer(1, 4), pipeline_drawing(W), Paragraph("Figure 1 — the pipeline from facts to outputs.",
                                                           ST["caption"])]
    story += [H2("Conventions used everywhere", "s1c")]
    story += bullets([
        "SI units. Angles are degrees in `.env` and in the API (`tilt_deg`), radians inside formulas.",
        "Lander frame (TN D-7959 axes): origin on the lander's centre line at the footpad plane, +Z up the thrust "
        "axis, leg 1 on +Y. The centre of gravity sits on this centre line, 0.79 m up, and never moves.",
        "World frame: origin at the landing target on the ground, +Z up. The ground is a plane through the origin, "
        "tilted for slope runs.",
        "MuJoCo state: `qpos` (7) = lander frame origin (3, world) and attitude quaternion (4, w x y z). `qvel` (6) "
        "= origin velocity (3, world) and angular velocity (3, body axes). The legs are rigid, so there are no "
        "joints.",
        "Throttle = Viking valve setting u in [0, 1]: thrust = 276 N + u × (2,667 − 276) N, so u = 0 is minimum "
        "thrust, not off. Engines stop only through the engines-off command (permanent) or an empty tank.",
        "Determinism (D10): one seed spawns two random streams (engine misalignment, radar noise) in a fixed "
        "order, and MuJoCo is deterministic, so the same seed, config and commands reproduce an episode bit for "
        "bit on the same machine.",
        "The vehicle layer (`config`, `geometry`, `vehicle`, `aero`, `sensors`, `touchdown`) is plain numpy and "
        "never imports MuJoCo (D1); `sim.py` connects it to MuJoCo.",
    ])
    story += [CondPageBreak(330), Spacer(1, 4), step_flow_drawing(W),
              Paragraph("Figure 2 — one control step: the data passed between the stages of LanderSim.step and "
                        "LanderSim._physics_step.", ST["caption"])]

    # ---------------------------------------------------------------- 2 how to read
    story += [CondPageBreak(260), Spacer(1, 6), H1("2  How to read the tree", "s2")]
    story += bullets([
        "A node is a function, method, property or nested function. A few nodes are not functions: the root "
        "`main` (the program entry), the test suite and its files, the notebook, and calls into external "
        "libraries (MuJoCo, imageio).",
        "A node's children are what it calls, in the order it calls them.",
        "A function called from several places is explained once, at its home position. Every other call appears "
        "as a reference node ([[REF]]) that links to the home and states the data passed at that call site.",
        "Every parent → child edge was checked against the source code by the script that made this document, "
        "and the file:line next to each node comes from the code itself.",
    ])
    story += [H2("In-order (LVR) for nodes with any number of children", "s2a"),
              para("In-order visits a binary node as Left subtree, then the node (Visit), then the Right subtree. The "
                   "functions here call any number of others, so in-order is applied in its usual general-tree form: "
                   "L = the subtree of the first child, V = the node, R = the subtrees of the remaining children, "
                   "left to right. The number in brackets on every node is its position in that walk, and Section 5 "
                   "explains the nodes in exactly that numeric order. Read this way, each function comes right after "
                   "the first thing it calls and right before the later steps it coordinates."),
              lvr_drawing(W),
              Paragraph("Figure 3 — in-order on a small tree: a1, A, P, B, c1, C, c2.", ST["caption"]),
              H2("What each entry in Section 5 contains", "s2b")]
    story += bullets([
        "Header: in-order number, name with parameters, kind and file:line.",
        "Parent, L and R: links to the caller, the first callee and the other callees.",
        "What it does.",
        "Interface: **Receives** = the data its parent passes in; **Returns** = what goes back to the parent; "
        "**State** = data exchanged through objects instead of arguments (MuJoCo arrays, cached fields); "
        "**Calls** = for each child, the data passed down and the value that comes back.",
        "Reference ([[REF]]) and external nodes get a one-line entry with the data of that call.",
        "Every [number] is a link. The PDF's bookmark panel lists all nodes too.",
    ])

    # ---------------------------------------------------------------- 3 data types
    story += [PageBreak(), H1("3  The data that travels between nodes", "s3"),
              para("These records are the vocabulary of every interface in Section 5."),
              H2("Config (from .env)", "s3cfg"),
              para("`Config` holds one frozen dataclass per key prefix, plus `params` (key → Param) and `env_path`. "
                   "Field name = key minus prefix, lower case: `VIKING_ENGINE_THRUST_MAX_N` → "
                   "`cfg.engine.thrust_max_n`. `SimCfg` adds two properties, `control_dt_s` (0.04 s) and `substeps` "
                   "(20).")]
    rows = [["prefix", "group", "keys", "examples"]]
    for prefix, name, cls in GROUPS:
        keys = [k for k in cfg.params if k.startswith(prefix)
                and not any(k.startswith(o) for o, _, _ in GROUPS if o != prefix and o.startswith(prefix))]
        rows.append([prefix + "*", f"cfg.{name}", str(len(keys)), ", ".join(k[len(prefix):].lower() for k in keys[:4])
                     + (" …" if len(keys) > 4 else "")])
    story += [table(rows, [100, 70, 30, 316], mono_first=True), Spacer(1, 6)]
    story += type_tables()
    story += [H2("MuJoCo objects", "s3mj")]
    story += bullets([
        "`MjModel` (model): constants compiled from the generated XML. The simulator writes the lander's "
        "`body_mass` and `body_subtreemass` as propellant burns (the CG and inertia never change), and the "
        "terrain's `geom_friction` at reset.",
        "`MjData` (data): the state. `qpos`, `qvel` (layout in Section 1), `time`, `mocap_quat` (ground tilt), "
        "`xfrc_applied` (force and torque on the lander body, written every physics step), `contact` (read by "
        "`_contacts`), `sensordata` (IMU accelerometer and gyro at the CG).",
    ])

    # ---------------------------------------------------------------- 4 tree
    story += [PageBreak(), H1("4  The logical tree", "s4"),
              para("Figure 4 shows the top three levels. The complete tree follows: every node with its in-order "
                   "number (a link to its entry), the file and line where it is defined, and "
                   "[[REF]] for references to a node explained elsewhere."),
              overview_drawing(W),
              Paragraph("Figure 4 — main(), the entry points and what each calls first.", ST["caption"]),
              PageBreak(), H2("The complete call tree", "s4b"), outline_rows()]

    # ---------------------------------------------------------------- 5 entries
    story += [PageBreak(), H1("5  Every node, in-order (LVR)", "s5"),
              para(f"Nodes [1] to [{n_nodes}] in in-order. Full entries for functions, compact ones for "
                   "references and external calls.")]
    for n in ORDER:
        if n.kind in ("ref", "external"):
            story.append(compact_entry(n))
        else:
            story.append(full_entry(n))

    # ---------------------------------------------------------------- appendices
    story += [PageBreak(), H1("Appendix A  Functions on no call path from main", "sa"),
              para("These are kept for the next milestones or as a small public API; nothing in the scripts calls "
                   "them yet (tests may).")]
    rows = [["function", "where", "what it is for"]]
    no_doc: dict[str, str] = {}   # one-line descriptions for unused functions without a docstring
    for key in sorted(UNUSED):
        info = INDEX[key]
        q = key.split("::")[1]
        text = (info["doc"] or "").split("\n")[0] or no_doc.get(q, "")
        if not text:
            raise SystemExit(f"describe unused function {q}")
        rows.append([q, f"{info['file']}:{info['line']}", text])
    story += [table(rows, [130, 110, 276], mono_first=True)]
    story += [H1("Appendix B  Where each file's functions are explained", "sb")]
    by_file: dict[str, list] = {}
    for n in ALL:
        if n.kind in SOURCE_KINDS:
            by_file.setdefault(n.src["file"], []).append(n)
    rows = [["file", "nodes (in-order number and name)"]]
    for file in sorted(by_file):
        nodes = sorted(by_file[file], key=lambda x: x.num)
        rows.append([file, ", ".join(f'{link(x)} {esc(label(x))}' for x in nodes)])
    t = Table([[Paragraph(fmt(r[0]) if i == 0 else f'<font name="Mono" size="7.6">{esc(r[0])}</font>',
                          ST["cellb"] if i == 0 else ST["cell"]),
                Paragraph(r[1] if i else fmt(r[1]), ST["cellb"] if i == 0 else ST["cell"])]
               for i, r in enumerate(rows)], colWidths=[128, 388], repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
                           ("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]))
    story += [t]

    check_glyphs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = GuideDoc(OUT)
    doc.multiBuild(story)
    print(f"wrote {OUT}  nodes={n_nodes} source={n_src} refs={n_ref} ext={n_ext} unused={len(UNUSED)}")


if __name__ == "__main__":
    build()
