import json
import subprocess
from pathlib import Path

from hwsol.adapters.base import RenderError, RenderResult
from hwsol.config import RenderConfig


class MarkdownToHandwriteAdapter:
    """Run the compatibility backend in its own environment (Pillow <11)."""

    name = "markdown-to-handwrite"

    def __init__(self, python: Path):
        # Resolving a venv's Python symlink loses the venv's site-packages.
        self.python = python.expanduser().absolute()

    def render(self, source: str, font: Path, config: RenderConfig, output: Path) -> RenderResult:
        from handnote.coverage import analyze, document_chars, font_codepoints
        from handnote.markdown import parse
        from matplotlib import font_manager

        # Upstream can silently omit Greek letters when every font lacks a glyph.
        fallback = Path(font_manager.findfont("STIXGeneral", fallback_to_default=False))
        text, math, _ = document_chars(parse(source))
        covered = font_codepoints(font) | font_codepoints(fallback)
        missing = sorted(ch for ch in text | math if ord(ch) not in covered)
        if missing:
            raise RenderError(
                f"Glyphs missing from selected and fallback fonts: {''.join(missing)}"
            )
        worker = Path(__file__).with_name("_mth_worker.py")
        request = {
            "source": source,
            "font": str(font),
            "fallback_font": str(fallback),
            "config": config.model_dump(),
            "output": str(output),
        }
        try:
            proc = subprocess.run(
                [str(self.python), "-I", str(worker)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RenderError(f"Compatibility backend could not run: {exc}") from exc
        if proc.returncode:
            raise RenderError(f"Compatibility backend failed: {proc.stderr[-4000:]}")
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RenderError(f"Invalid backend response: {proc.stdout[-1000:]}") from exc
        result["coverage"] = analyze(font, parse(source)).to_dict()
        result["backend_details"]["fallback_font"] = str(fallback)
        return RenderResult(**result)
