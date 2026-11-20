"""Isolated worker: only stdlib and the compatibility environment are imported."""

import contextlib
import io
import json
import sys
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    from markdown_to_handwrite.config import config_from_dict
    from markdown_to_handwrite.markdown_parser import parse_markdown
    from markdown_to_handwrite.renderer import ReportRenderer

    data = json.load(sys.stdin)
    output = Path(data["output"])
    c = data["config"]
    config = config_from_dict(
        {
            "page": {
                "width_mm": c["page_width_mm"],
                "height_mm": c["page_height_mm"],
                "dpi": c["dpi"],
                **{
                    f"margin_{side}_mm": c["margin_mm"]
                    for side in ("left", "right", "top", "bottom")
                },
            },
            "background": {
                "style": "dot" if c["paper"] == "dots" else c["paper"],
                "auto_discover": False,
                "paper_color": "#ffffff",
                "draw_margin_line": False,
            },
            "handwriting": {
                "font_path": data["font"],
                "math_font_path": data["font"],
                "fallback_font_path": data["fallback_font"],
                "seed": c["seed"],
                "body_font_pt": c["font_size_pt"],
                "math_font_pt": c["font_size_pt"],
                "line_spacing": c["line_spacing"],
                "ink_color": c["ink"],
                "sdt_trajectory_enabled": False,
                # Even a disabled store otherwise opens the bundled archive in __init__.
                "sdt_trajectory_path": str(output / "disabled-trajectories-do-not-create"),
                "second_layer_enabled": False,
                "prefer_handright": False,
                "perturb_theta_sigma": 0,
                "perturb_x_sigma_px": 0,
                "perturb_y_sigma_px": 0,
                "math_perturb_y_sigma_ratio": 0,
                "advance_jitter_sigma_ratio": 0,
            },
            "layout": {"show_page_numbers": False, "first_line_indent_em": 0},
        }
    )
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
        renderer = ReportRenderer(config, base_dir=output)
        pages = renderer.render(parse_markdown(data["source"]))
    diagnostics = captured.getvalue()
    if "[latex-render-error]" in diagnostics:
        raise ValueError(diagnostics)
    renderer.save_pdf(output / "probe.pdf")
    (output / "pages").mkdir()
    for index, page in enumerate(pages, 1):
        page.convert("RGB").save(output / "pages" / f"{index:02d}.png", dpi=(c["dpi"], c["dpi"]))
    print(
        json.dumps(
            {
                "page_count": len(pages),
                "warnings": [
                    "Experimental compatibility backend: glyph fallback is not fully audited."
                ]
                + ([diagnostics] if diagnostics else []),
                "coverage": {"status": "not_available"},
                "backend_details": {
                    "version": version("MarkdownToHandwrite"),
                    "pdf_mode": "raster",
                    "sdt_trajectories_enabled": renderer.engine.uses_sdt_trajectories,
                    "font_chain": renderer.engine.font_paths,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
