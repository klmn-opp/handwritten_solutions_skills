"""Bind MathText to the snapshot on disk, including after an in-process Profile switch."""

from handnote.render import Renderer
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path as MplPath
from numpy import ndarray


class ProfileRenderer(Renderer):
    def _apply_mathtext(self) -> None:
        rcParams["mathtext.fontset"] = "custom"
        rcParams["mathtext.fallback"] = "stix"
        pattern = FontProperties(fname=str(self.font_path.resolve())).get_fontconfig_pattern()
        for face in ("rm", "it", "bf"):
            rcParams[f"mathtext.{face}"] = pattern
        # The pinned parser caches (tex, size, dpi, properties), omitting the custom faces.
        # Called inside the adapter's render lock at the start of each render pass.
        MathTextParser._parse_cached.cache_clear()

    def _math_glyph(
        self, font: FT2Font, index: int, ccode: int, k: int, handwritten: bool = False
    ) -> tuple[ndarray, ndarray]:
        vertices, codes = super()._math_glyph(font, index, ccode, k, handwritten)
        if not handwritten or k == 0:
            return vertices, codes
        # MathText has already positioned neighbours using the primary glyph's metrics.
        # A wider alternate must fit that width in the raster and vector paths alike.
        primary, primary_codes = super()._math_glyph(font, index, ccode, 0, True)
        base = primary[primary_codes != MplPath.CLOSEPOLY]
        ink = vertices[codes != MplPath.CLOSEPOLY]
        if not len(base) or not len(ink):
            return vertices, codes
        left, right = base[:, 0].min(), base[:, 0].max()
        lo, hi = ink[:, 0].min(), ink[:, 0].max()
        if hi <= lo or right <= left:
            return vertices, codes
        scale = min(1.0, (right - left) / (hi - lo))
        # Width fitting must not shrink the writer's digits vertically. Match the
        # primary's vertical metrics, which MathText used for accents and fractions.
        fitted = vertices.copy()
        fitted[:, 0] *= scale
        fitted[:, 0] += max(left - lo * scale, min(0.0, right - hi * scale))
        bottom, top = base[:, 1].min(), base[:, 1].max()
        low, high = ink[:, 1].min(), ink[:, 1].max()
        if high > low and top > bottom:
            fitted[:, 1] = bottom + (vertices[:, 1] - low) * (top - bottom) / (high - low)
        return fitted, codes
