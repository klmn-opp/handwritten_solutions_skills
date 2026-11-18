from pathlib import Path
from threading import RLock

from hwsol.adapters.base import RenderError, RenderResult
from hwsol.config import RenderConfig

RENDER_LOCK = RLock()


def upstream_config(font: Path, config: RenderConfig, real_variants: bool):
    from handnote.config import RenderConfig as UpstreamConfig

    size = config.pt(config.font_size_pt)
    return UpstreamConfig.from_mapping(
        {
            "font_path": str(font),
            "page_width": config.mm(config.page_width_mm),
            "page_height": config.mm(config.page_height_mm),
            "margin_x": config.mm(config.margin_mm),
            "margin_top": config.mm(config.margin_mm),
            "margin_bottom": config.mm(config.margin_mm),
            "body_size": size,
            "display_size": size,
            "h1_size": round(size * 1.3),
            "h2_size": round(size * 1.15),
            "h3_size": size,
            "line_height": config.line_spacing,
            "dpi": config.dpi,
            "seed": config.seed,
            "ink": config.ink,
            "paper": True,
            "paper_color": "#ffffff",
            "paper_style": config.paper,
            "paper_gradient": 0,
            "paper_noise": 0,
            "page_numbers": False,
            # Two enables upstream's real GSUB alternates; strength zero disables warps.
            "glyph_sets": 2 if real_variants else 1,
            "glyph_strength": 0,
            "jitter_rot": 0,
            "jitter_dx": 0,
            "jitter_dy": 0,
            "size_wobble": 0,
            "ink_alpha_min": 255,
            "ink_alpha_max": 255,
        }
    )


class HandnoteAdapter:
    name = "handnote"

    def __init__(self, *, real_variants: bool = False):
        self.real_variants = real_variants

    def render(self, source: str, font: Path, config: RenderConfig, output: Path) -> RenderResult:
        from handnote.coverage import document_chars, font_codepoints
        from handnote.markdown import parse
        from handnote.render import Renderer, flatten
        from matplotlib import rc_context

        blocks = parse(source)
        if not blocks:
            raise RenderError("Markdown contains no renderable content")
        text_chars, _, _ = document_chars(blocks)
        covered = font_codepoints(font)
        missing_text = sorted(ch for ch in text_chars if ord(ch) not in covered)
        if missing_text:
            raise RenderError(f"Text glyphs missing from selected font: {''.join(missing_text)}")
        cfg = upstream_config(font, config, self.real_variants)
        with RENDER_LOCK, rc_context():
            renderer = Renderer(cfg)
            coverage = renderer.coverage(blocks)
            images = renderer.render(blocks)
            pdf = renderer.pdf(blocks, images)
            if renderer.warnings:
                raise RenderError("Upstream degraded rendering: " + "; ".join(renderer.warnings))
        (output / "probe.pdf").write_bytes(pdf)
        page_dir = output / "pages"
        page_dir.mkdir()
        for index, image in enumerate(images, 1):
            flatten(image).save(page_dir / f"{index:02d}.png", dpi=(config.dpi, config.dpi))
        warnings = []
        if coverage["missing"]:
            warnings.append(
                "Math glyphs absent from selected font; upstream uses STIX fallback. "
                "See coverage; this is not a per-glyph fallback audit."
            )
        return RenderResult(
            len(images),
            warnings,
            coverage,
            {
                "pdf_mode": "vector",
                "png_mode": "same upstream layout, separate raster pass",
                "neural_models": False,
                "real_glyph_variants": self.real_variants,
                "upstream_config": cfg.as_dict(),
            },
        )
