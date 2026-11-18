"""Measure Handnote's placed vector outlines before writing a one-problem PDF."""

import logging
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
from handnote.coverage import document_chars
from handnote.markdown import Block, parse
from handnote.render import Renderer
from matplotlib import rc_context
from matplotlib.path import Path as MplPath

from hwsol.adapters.handnote import RENDER_LOCK, upstream_config
from hwsol.config import RenderConfig
from hwsol.jobs.paper import paper_image, visible_ink
from hwsol.jobs.schema import Background
from hwsol.profiles.store import LoadedProfile

COMMANDS = frozenset(
    """
frac sqrt sum prod int iint oint lim to infty sin cos tan log ln exp min max
vec hat bar overline dot ddot partial nabla mathrm mathbf mathit mathbb
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda
mu nu xi pi rho sigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda
Xi Pi Sigma Upsilon Phi Psi Omega times cdot div pm mp le leq ge geq ne neq approx
equiv sim propto in notin subset subseteq cup cap forall exists left right
langle rangle lvert rvert vert Vert ldots cdots dots quad qquad begin end
""".split()
)
_COMMAND = re.compile(r"\\([a-zA-Z]+|.)")
_ENV = re.compile(r"\\(?:begin|end)\{([^{}]+)\}")


class MultiplePages(Exception):
    def __init__(self, count):
        self.count = count


class SinglePageRenderer(Renderer):
    def paginate(self, lines):
        pages = super().paginate(lines)
        if len(pages) != 1:
            raise MultiplePages(len(pages))
        return pages


class LayoutError(ValueError):
    def __init__(self, attempts):
        self.attempts = attempts
        super().__init__(
            "Solution does not fit one page. Rewrap equations or remove repeated prose."
        )


def validate_markdown(source: str, profile: LoadedProfile):
    if not source.strip():
        raise ValueError("Empty solution")
    for line in source.splitlines():
        # Display delimiters pair across lines. Inline math must close on its own line.
        rest = re.sub(r"(?<!\\)\$\$", "", line)
        if len(re.findall(r"(?<!\\)\$", rest)) % 2:
            raise ValueError("Unclosed inline math delimiter")
        if "$$" in line and not (line.strip().startswith("$$") or line.strip().endswith("$$")):
            raise ValueError("Place display math on separate lines")
    for name in _COMMAND.findall(source):
        if name not in COMMANDS and name not in {
            ",",
            ";",
            "!",
            " ",
            "\\",
            "{",
            "}",
            "%",
            "$",
            "|",
            ":",
        }:
            raise ValueError(f"Unsupported LaTeX command: {name}")
    if any(env not in {"matrix", "pmatrix", "vmatrix", "bmatrix"} for env in _ENV.findall(source)):
        raise ValueError("Unsupported LaTeX environment")
    blocks = parse(source)
    if not any(
        b.kind in {"matrix", "pagebreak"} or any(v.strip() for _, v in b.segments) for b in blocks
    ):
        raise ValueError("Empty solution")
    if any(b.kind == "pagebreak" for b in blocks):
        raise ValueError("A solution cannot contain an explicit page break")
    chars, _, _ = document_chars(blocks)
    for block in blocks:
        # Handnote uses the text font directly for single-character standalone matrix cells.
        if block.kind == "matrix":
            chars.update(cell for row in block.payload["rows"] for cell in row if len(cell) == 1)
    missing = sorted(c for c in chars if ord(c) not in profile.manifest.coverage)
    if missing:
        raise ValueError(f"Missing text or matrix glyphs in Profile: {''.join(missing)}")
    return blocks


def compact(blocks):
    """Break display equality chains only at top-level equality signs."""
    result = []
    for block in blocks:
        if block.kind == "blank":
            continue
        if block.kind != "display":
            result.append(block)
            continue
        expr = block.segments[0][1]
        depth, cuts = 0, []
        for pos, c in enumerate(expr):
            if c in "{([":
                depth += 1
            elif c in "})]":
                depth -= 1
            elif c == "=" and depth == 0:
                cuts.append(pos)
        if len(cuts) < 2 or r"\begin" in expr or r"\left" in expr:
            result.append(block)
            continue
        boundaries = [0, *cuts[1:], len(expr)]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            result.append(Block("display", (("m", expr[start:end]),), block.payload))
    return result


def ink_geometry(page, renderer):
    """Use the pinned recorder's item/outline format, also used by its PDF encoder."""
    lo = np.array([math.inf, math.inf])
    hi = -lo
    math_fonts = set()
    for item in page.items:
        kind, shape, matrix = item[:3]
        if kind == "path":
            outline = renderer._outlines[shape]
            points = outline.verts[outline.codes != MplPath.CLOSEPOLY]
            width = item[4] if outline.stroke else 0
            if shape[0] == "mglyph":
                if not shape[2]:
                    raise ValueError("Formula contains a missing glyph")
                math_fonts.add((int(shape[3]), str(shape[1])))
            if shape[0] == "math":
                raise ValueError("Formula backend did not expose auditable glyphs")
        elif kind == "stroke":
            points, width = np.asarray(shape), item[4]
        elif kind == "image":
            box = shape.getchannel("A").getbbox()
            if box is None or min(box[:2]) <= 0 or box[2] >= shape.width or box[3] >= shape.height:
                raise ValueError("Raster glyph is empty or touches its tile boundary")
            x0, y0, x1, y1 = box
            points, width = np.array([(x0, y0), (x1, y0), (x0, y1), (x1, y1)]), 0
        else:
            raise ValueError(f"Unsupported ink recording: {kind}")
        if not len(points):
            continue
        a, b, c, d, e, f = matrix
        coords = points @ np.array([[a, b], [c, d]]) + [e, f]
        padding = width * np.array([math.hypot(a, c), math.hypot(b, d)]) / 2
        lo = np.minimum(lo, coords.min(axis=0) - padding)
        hi = np.maximum(hi, coords.max(axis=0) + padding)
    if not np.isfinite([lo, hi]).all():
        raise ValueError("Empty or invalid ink bounds")
    return [*lo.tolist(), *hi.tolist()], sorted(math_fonts)


class MathDiagnostics(logging.Handler):
    def emit(self, record):
        if "dummy symbol" in record.getMessage():
            raise ValueError(record.getMessage())


def render_one(
    source: str,
    number: str,
    profile: LoadedProfile,
    cfg: RenderConfig,
    bg: Background,
    asset: Path | None,
):
    blocks = validate_markdown(source, profile)
    title = Block("h2", (("t", number),))
    if any(ord(c) not in profile.manifest.coverage for c in number if not c.isspace()):
        raise ValueError(f"Missing glyph in problem number {number}")
    # Work from the same original blocks for every attempt. Markdown on disk remains editable.
    options = [
        (False, False, cfg.margin_mm, cfg.font_size_pt, cfg.line_spacing),
        (True, False, cfg.margin_mm, cfg.font_size_pt, cfg.line_spacing),
        (True, True, cfg.margin_mm, cfg.font_size_pt, cfg.line_spacing),
        (True, True, min(cfg.margin_mm, 12), cfg.font_size_pt, cfg.line_spacing),
        (
            True,
            True,
            min(cfg.margin_mm, 12),
            min(cfg.font_size_pt, 11),
            min(cfg.line_spacing, 1.35),
        ),
        (True, True, min(cfg.margin_mm, 10), min(cfg.font_size_pt, 10), min(cfg.line_spacing, 1.2)),
    ]
    attempts = []
    for trim, rewrap, margin, size, spacing in dict.fromkeys(options):
        body = [b for b in blocks if not trim or b.kind != "blank"]
        if rewrap:
            body = compact(body)
        settings = cfg.model_copy(
            update=dict(
                margin_mm=margin,
                font_size_pt=size,
                line_spacing=spacing,
                paper=bg.kind if asset is None else "plain",
            )
        )
        upstream = upstream_config(profile.font, settings, bool(profile.manifest.variants))
        record = {
            "margin_mm": margin,
            "font_size_pt": size,
            "line_spacing": spacing,
            "strip_blank_blocks": trim,
            "rewrap_equalities": rewrap,
        }
        attempts.append(record)
        with RENDER_LOCK, rc_context():
            logger = logging.getLogger("matplotlib._mathtext")
            diagnostic = MathDiagnostics()
            logger.addHandler(diagnostic)
            try:
                renderer = SinglePageRenderer(upstream)
                pages = renderer.render_vector([title, *body])
            except MultiplePages as exc:
                record["page_count"] = exc.count
                continue
            finally:
                logger.removeHandler(diagnostic)
            if renderer.warnings:
                raise ValueError("; ".join(renderer.warnings))
            record["page_count"] = len(pages)
            if len(pages) != 1:
                continue
            page = pages[0]
            bounds, used = ink_geometry(page, renderer)
            record["ink_bounds_px"] = bounds
            # Allow two pixels of glyph overhang at the requested typesetting margin.
            edge = max(1, settings.mm(margin) - 2)
            if (
                min(bounds[:2]) < edge
                or bounds[2] > page.width - edge
                or bounds[3] > page.height - edge
            ):
                record["reason"] = "ink crosses content boundary"
                continue
            if asset is not None:
                page.background = paper_image(bg, asset, (page.width, page.height))
                page.paper = None
            fallback = [
                {"codepoint": f"U+{cp:04X}", "font": Path(font).name}
                for cp, font in used
                if Path(font).resolve() != profile.font.resolve()
            ]
            pdf = renderer.to_vector_pdf(pages)
            paper_only = renderer.to_vector_pdf([replace(page, items=[])])
            if not visible_ink(pdf, paper_only):
                raise ValueError("Ink is not visible against the selected paper")
            return pdf, {
                "attempts": attempts,
                "math_fallback_glyphs": fallback,
                "coverage": renderer.coverage(body),
            }
    raise LayoutError(attempts)
