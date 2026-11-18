import json
import os
from pathlib import Path

import pytest

from hwsol.adapters.base import RenderError
from hwsol.adapters.markdown_to_handwrite import MarkdownToHandwriteAdapter
from hwsol.config import RenderConfig
from hwsol.config import test_font as bundled_test_font
from hwsol.probe import run_probe

pytestmark = pytest.mark.skipif(
    not os.environ.get("HWSOL_MTH_PYTHON"), reason="Optional isolated compatibility environment"
)


@pytest.mark.parametrize("valid", [True, False])
def test_actual_isolated_backend(tmp_path, valid):
    source = tmp_path / "source.md"
    source.write_text(
        r"$$\alpha+\beta=\sigma$$" if valid else r"$$\notARealCommand{x}$$", encoding="utf-8"
    )
    adapter = MarkdownToHandwriteAdapter(Path(os.environ["HWSOL_MTH_PYTHON"]))
    output = tmp_path / "output"
    if not valid:
        with pytest.raises(RenderError, match="latex-render-error"):
            run_probe(source, output, bundled_test_font(), RenderConfig(), adapter)
        assert not (output / "probe.pdf").exists()
        return
    report = run_probe(source, output, bundled_test_font(), RenderConfig(), adapter)
    details = report["backend_details"]
    assert report["page_count"] == 1
    assert details["sdt_trajectories_enabled"] is False
    assert "STIXGeneral.ttf" in details["fallback_font"]
    assert details["fallback_font"] in details["font_chain"]
    assert json.loads((output / "validation_report.json").read_text())["status"] == "passed"
