"""CPU planning on top of pinned Handnote tokenization, metrics and vector recording."""

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
from handnote.markdown import Block
from handnote.render import _drop_math_spaces, _tokenize
from handnote.variants import SetPicker
from matplotlib.path import Path as MplPath

from hwsol.jobs.ink_policy import PolicyInk, fitted_variant
from hwsol.profiles.store import LoadedProfile
from hwsol.profiles.style import LineStyle, PageStyle


def split_simple_chain(block: Block) -> list[Block]:
    """Fold only a flat arithmetic chain; leave every compound TeX expression intact."""
    expr = block.segments[0][1]
    if not re.fullmatch(r"[ A-Za-z0-9.+*/=\-]+", expr):
        return [block]
    terms = expr.split("=")
    if len(terms) < 3 or not all(term.strip() for term in terms):
        return [block]
    parts = [terms[0] + "=" + terms[1], *["=" + term for term in terms[2:]]]
    return [Block("display", (("m", part),), block.payload) for part in parts]


class HandwritingLayoutPlanner:
    """Plan in em/page fractions so calibration transfers between paper sizes and DPI."""

    def __init__(
        self, renderer, sample: PageStyle | None, rng: random.Random, styles: tuple[PageStyle, ...]
    ):
        self.renderer = renderer
        cfg = renderer.cfg
        self.line_styles = {}
        self.line_sources = {}
        self.used_sources = {}
        for kind in ("text", "math"):
            sources = [s for s in styles if any(line.kind == kind for line in s.lines)]
            if sources:
                source = sample if sample in sources else rng.choice(sources)
                self.line_styles[kind] = [line for line in source.lines if line.kind == kind]
                self.line_sources[kind] = source.sample_id
        # A readable baseline can be annotated even when an insertion obscures word boundaries.
        measured_gaps = [g for row in self.line_styles.get("text", ()) for g in row.word_gaps_em]
        self.default_word_gap = float(np.median(measured_gaps)) if measured_gaps else 0.28
        self.rng = rng
        self.left = cfg.margin_x
        self.right = cfg.page_width - cfg.margin_x
        if sample:
            self.right = min(self.right, cfg.page_width * (1 - min(0.3, sample.right)))
        self.recent = rng.uniform(0.05, 0.35)
        # Per-page cache, discarded with the renderer after each fit attempt.
        self.kern = lru_cache(maxsize=2048)(self._kern)

    def _kern(self, pair: str, size: int) -> float:
        font = self.renderer.font(size)
        adjustment = font.getlength(pair) - sum(font.getlength(ch) for ch in pair)
        return max(-0.06 * size, min(0.06 * size, adjustment))

    def rhythm(self, kind: str, index: int) -> LineStyle:
        if matching := self.line_styles.get(kind):
            self.used_sources[kind] = self.line_sources[kind]
            return matching[index % len(matching)]
        # Bounded, correlated motion: neighbouring baseline knots share their history.
        self.recent = 0.72 * self.recent + 0.28 * self.rng.random()
        knots = [0.0]
        for _ in range(4):
            knots.append(0.55 * knots[-1] + self.rng.uniform(-0.075, 0.075))
        return LineStyle(
            kind=kind,
            anchor=0.10 + self.recent * 0.12,
            inset=(self.left + self.recent * self.renderer.cfg.body_size)
            / self.renderer.cfg.page_width,
            pitch_em=None,
            baseline_em=tuple(knots),
            word_gaps_em=tuple(0.31 + 0.09 * self.recent + 0.6 * offset for offset in knots),
        )

    def space_advance(self, target: float, left_token, right_token, size: int) -> float:
        # Calibration measures visible ink-to-ink gaps. Side bearings already add whitespace.
        r = self.renderer
        before = after = 0.0
        if left_token and left_token[0] == "t":
            ch = left_token[1][-1]
            ink = r._ink(ch, size)
            if ink:
                shift, advance = r.char_metrics(ch, size)
                before = advance - shift - ink[1]
        if right_token and right_token[0] == "t":
            ch = right_token[1][0]
            ink = r._ink(ch, size)
            if ink:
                after = r.char_metrics(ch, size)[0] + ink[0]
        return max(0.05 * size, target - before - after)

    def text_lines(self, block, first_index: int) -> list[dict]:
        r = self.renderer
        size = r.heading_size(block.kind) if block.kind in {"h1", "h2", "h3"} else r.cfg.body_size
        tokens = _drop_math_spaces(_tokenize(block.segments))
        result = []
        start = 0
        while start < len(tokens):
            rhythm = self.rhythm("text", first_index + len(result))
            left = max(self.left, min(self.right - size * 2, rhythm.inset * r.cfg.page_width))
            if block.kind == "bullet":
                left += 0.6 * size
            cursor = left
            runs = []
            for index in range(start, len(tokens)):
                kind, word = tokens[index]
                if kind == "m":
                    width = r.math_advance(word, size)[1]
                    adjustments = ()
                elif word.isspace():
                    gaps = rhythm.word_gaps_em or (self.default_word_gap,)
                    gap = gaps[len(runs) // 2 % len(gaps)]
                    after_stop = bool(runs and runs[-1][1].endswith((".", "!", "?", ":", ",")))
                    target = size * gap * (1.16 if after_stop else 1.0)
                    following = tokens[index + 1] if index + 1 < len(tokens) else None
                    width = self.space_advance(target, runs[-1] if runs else None, following, size)
                    adjustments = ()
                else:
                    adjustments = tuple(self.kern(a + b, size) for a, b in pairwise(word))
                    width = sum(r.char_metrics(c, size)[1] for c in word) + sum(adjustments)
                if runs and cursor + width > self.right and not word.isspace():
                    break
                runs.append((kind, word, adjustments, width))
                cursor += width
            line = r._text_line(
                block.kind, size, left - self.left, [(kind, word) for kind, word, _, _ in runs]
            )
            pitch = rhythm.pitch_em or (r.cfg.line_height + self.rng.uniform(-0.07, 0.12))
            # Keep tall inline fractions/matrices and the full baseline envelope in the line box.
            excursion = max(abs(v) for v in rhythm.baseline_em) * size
            line["h"] += (max(1.35, pitch) - r.cfg.line_height) * size + 2 * excursion
            line["pad"] += excursion
            line.update(runs=runs, curve=rhythm.baseline_em, span=max(size, cursor - left))
            result.append(line)
            start += len(runs)
        return result

    def plan(self, blocks) -> list[dict]:
        r = self.renderer
        result = []
        text_rows = math_rows = 0
        for block in blocks:
            if block.kind in {"h1", "h2", "h3", "p", "num", "bullet"} and not block.payload:
                rows = self.text_lines(block, text_rows)
                text_rows += len(rows)
                result.extend(rows)
                if block.kind in {"h1", "h2"}:
                    result.append({"kind": "gap", "h": r.cfg.body_size * 0.22})
                continue
            pieces = [block]
            if block.kind == "display" and not block.payload:
                tex = block.segments[0][1]
                if r.math_image(tex, r.cfg.display_size)[0].width > self.right - self.left:
                    pieces = split_simple_chain(block)
            lines = r.original_layout(pieces)
            for line in lines:
                kind = line["kind"]
                if kind in {"matrix", "display"} and not (
                    {"align", "indent"} & block.payload.keys()
                ):
                    rhythm = self.rhythm("math", math_rows)
                    math_rows += 1
                    width = (
                        line["img"].width
                        if kind == "display"
                        else r.matrix_width(line["prefix"], line["rows"])
                    )
                    x = self.left + max(0, self.right - self.left - width) * rhythm.anchor
                    if "math" in self.line_styles:
                        x = max(
                            self.left,
                            min(self.right - width, rhythm.anchor * (r.cfg.page_width - width)),
                        )
                    if kind == "display":
                        line["x"] = x
                    else:
                        line["cx"] = x + width / 2
                    if rhythm.pitch_em:
                        line["h"] = max(line["h"], rhythm.pitch_em * r.cfg.body_size)
            result.extend(lines)
        return result

    def draw_line(self, line, top, state):
        r = self.renderer
        size = line["size"]
        origin = x = self.left + line["indent"]
        baseline = top + line["pad"] + size * 1.28
        knots = line["curve"]

        def level(x_pos):
            t = max(0.0, min(4.0, 4 * (x_pos - origin) / line["span"]))
            i = min(3, int(t))
            f = t - i
            f = f * f * (3 - 2 * f)
            return baseline + size * ((1 - f) * knots[i] + f * knots[i + 1])

        for kind, word, adjustments, width in line["runs"]:
            if kind == "m":
                px = round(size * r.cfg.math_scale)
                _, base = r.math_image(word, px)
                shift, _ = r.math_advance(word, size)
                r._place_math(None, word, px, (round(x + shift), int(level(x) - base)))
                x += width
            elif word.isspace():
                x += width
            else:
                for i, ch in enumerate(word):
                    x += r.draw_char(None, x, level(x), ch, size, state)
                    if i < len(adjustments):
                        x += adjustments[i]
        return {
            "kind": line["kind"],
            "x": round(origin, 2),
            "baseline": round(baseline, 2),
            "baseline_knots_em": knots,
        }


def glyph_info(shape: tuple, profile: LoadedProfile, font: Path) -> tuple[int, str, int] | None:
    if shape[0] == "mglyph":
        code, slot = shape[3], shape[4]
        if Path(shape[1]).resolve() != font:
            return code, "fallback", 0
    elif shape[0] == "glyph":
        slot, code = shape[1:3]
    else:
        return None
    source = "personal" if code in profile.manifest.personal_codepoints else "base"
    return code, source, max(0, slot) if str(code) in profile.manifest.variants else 0


def finish_ink(page, renderer, profile, strokes, seed):
    """Fill the writer's outlines once; fit same-codepoint alternates in their measured box."""
    rng = random.Random(seed)
    picker = SetPicker(1, rng)
    font_path = profile.font.resolve()
    lean = renderer.style.pen_lean if renderer.style else None
    # Small shared bias on text letterforms, bounded inside each original ink box.
    lean = rng.uniform(0.055, 0.10) if lean is None else float(np.clip(lean, -0.18, 0.18))
    selector = (
        PolicyInk(renderer.policy_config, profile, seed)
        if renderer.policy_config is not None
        else None
    )

    info_cache = {}
    audit = Counter()
    for i, item in enumerate(page.items):
        if item[0] != "path":
            audit[("geometry" if item[0] == "stroke" else "raster", "", "", "", 0)] += 1
            continue
        _, shape, matrix, alpha, _ = item
        # A stroked PDF path of width zero still paints a device hairline. Disable the
        # stroke operator itself, including for fallback math glyphs and fraction rules.
        outline = renderer._outlines[shape]
        if outline.stroke:
            outline = replace(outline, stroke=False)
            renderer._outlines[shape] = outline
        page.items[i] = ("path", shape, matrix, alpha, 0)
        if shape not in info_cache:
            info_cache[shape] = glyph_info(shape, profile, font_path)
        info = info_cache[shape]
        if info is None:
            if shape[0] == "personal-math-stroke":
                audit[
                    ("personal", shape[1], "U+0304" if shape[1] == "bar" else "", shape[2], 0)
                ] += 1
            else:
                audit[("geometry", shape[0], "", "", 0)] += 1
            continue
        code, source, slot = info
        if strokes and source != "fallback":
            choices = profile.manifest.variants.get(str(code), 1)
            learned = selector is not None and not selector.action.legacy
            slot = (
                selector.choose(shape, outline, font_path, code, choices)
                if learned
                else picker.pick(code, choices)
            )
            local_lean = selector.lean() if learned else lean
            key = (*shape, "pen", slot)
            if learned and shape[0] == "glyph" and chr(code).isalpha():
                local_lean = round(local_lean, 4)
                key = (*key, local_lean)
            if key not in renderer._outlines:
                points = outline.verts[outline.codes != MplPath.CLOSEPOLY]
                low, high = points.min(axis=0), points.max(axis=0)
                if choices > 1:
                    outline = fitted_variant(outline, shape, font_path, code, slot)
                if shape[0] == "glyph" and chr(code).isalpha() and high[0] > low[0]:
                    vertices = outline.verts.copy()
                    keep = outline.codes != MplPath.CLOSEPOLY
                    tilted = vertices[keep, 0] + (vertices[keep, 1] - low[1]) * local_lean
                    extent = float(np.ptp(tilted))
                    if extent > 0:
                        vertices[keep, 0] = (
                            low[0] + (tilted - tilted.min()) * (high[0] - low[0]) / extent
                        )
                        outline = replace(outline, verts=vertices)
                renderer._outlines[key] = outline
            page.items[i] = ("path", key, matrix, alpha, 0)
            if selector is not None:
                selector.observe(renderer._outlines[key], shape, code, slot, choices)
        route = "math" if shape[0] == "mglyph" else "text-font"
        font = Path(shape[1]).name if route == "math" else font_path.name
        audit[(source, route, f"U+{code:04X}", font, slot)] += 1
    renderer.policy_report = selector.report() if selector else None
    return [
        dict(source=s, route=r, codepoint=c, font=f, variant=k, count=n)
        for (s, r, c, f, k), n in sorted(audit.items())
    ]


class PlannedRendering:
    """Mixin for the existing one-page vector renderer. Explicit paper grids keep their rulings."""

    def configure(self, mode, profile, identity, policy=None):
        self.policy_config = policy
        seed = f"{self.cfg.seed}|{identity}"
        self.layout_enabled = mode in {"layout-enhanced", "combined"} and not self.grid.snap
        self.stroke_enabled = mode in {"stroke-enhanced", "combined"}
        self.stroke_seed = seed + "|stroke"
        layout_styles = tuple(style for style in profile.styles if style.lines)
        self.style = (
            random.Random(seed + "|sample").choice(layout_styles) if layout_styles else None
        )
        self.planner = HandwritingLayoutPlanner(
            self, self.style, random.Random(seed + "|layout"), profile.styles
        )
        self.rows = []
        self.formulas = []
        self.matrices = []
        if self.layout_enabled and self.style:
            self.grid = replace(
                self.grid,
                top=max(
                    self.grid.top,
                    min(self.cfg.page_height * 0.25, self.cfg.page_height * self.style.top),
                ),
            )

    def _place_math(self, canvas, tex, px, pos):
        start = len(self._rec)
        super()._place_math(canvas, tex, px, pos)
        # Fingerprint actual placed glyph identities, rules and local transforms before nib styling.
        # Moving the same formula anywhere on a page must preserve this digest.
        items = []
        for item in self._rec[start:]:
            kind, shape, (a, b, c, d, x, y) = item[:3]
            if kind == "path" and shape[0] == "mglyph":
                shape = (shape[0], Path(shape[1]).name, shape[2], shape[3])
            matrix = [round(v, 7) for v in (a, b, c, d, x - pos[0], y - pos[1])]
            items.append((kind, shape, matrix))
        digest = hashlib.sha256(json.dumps(items).encode()).hexdigest()
        self.formulas.append(
            {
                "latex": tex,
                "size_px": self.cfg.body_size,
                "position_px": pos,
                "items": len(items),
                "geometry_sha256": digest,
            }
        )

    def draw_matrix(self, canvas, prefix, rows, env, x_center, top, state):
        start = len(self._rec)
        super().draw_matrix(canvas, prefix, rows, env, x_center, top, state)
        self.matrices.append(
            {
                "environment": env,
                "prefix": prefix,
                "rows": rows,
                "items": len(self._rec) - start,
                "cell_and_matrix_size_px": self.matrix_metrics(rows),
            }
        )

    def original_layout(self, blocks):
        return super().layout(blocks)

    def layout(self, blocks):
        if self.layout_enabled:
            return self.planner.plan(blocks)
        return super().layout(blocks)

    def render_page(self, lines, page_no, total, state):
        if not self.layout_enabled:
            return super().render_page(lines, page_no, total, state)
        grid = self.grid
        top = grid.top
        try:
            for line in lines:
                if "runs" in line:
                    self.rows.append(self.planner.draw_line(line, top, state))
                elif line["kind"] not in {"gap", "blank"}:
                    self.grid = replace(grid, top=top)
                    super().render_page([line], page_no, total, state)
                    self.rows.append(
                        {
                            "kind": line["kind"],
                            "top": round(top, 2),
                            "x" if "x" in line else "cx": line.get("x", line.get("cx")),
                        }
                    )
                top += line["h"]
        finally:
            self.grid = grid
