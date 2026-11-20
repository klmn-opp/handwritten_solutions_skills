"""Safety boundaries for the isolated complete-homework comparison."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from hwsol.config import RenderConfig
from hwsol.jobs.schema import Background
from research.homework_benchmark import render
from research.homework_benchmark.render import claim_holdout, pixel_boundary_checks


def test_candidate_cannot_hide_english_changes_inside_a_math_slot():
    a = Image.new("RGB", (30, 30), "white")
    b = a.copy()
    b.putpixel((12, 12), (0, 0, 0))
    text = {"item_index": 0, "context": "text", "bounds_px": [12, 12, 15, 15]}
    math = {"item_index": 1, "context": "mathematics", "bounds_px": [5, 5, 8, 8]}
    changed_math = {**math, "bounds_px": [5, 5, 15, 15], "allowed_bounds_px": [4, 4, 16, 16]}
    with pytest.raises(ValueError, match="protected English"):
        pixel_boundary_checks(a, b, [text, math], [text, changed_math], {1})


def test_candidate_cannot_change_unassigned_pixels():
    a = Image.new("RGB", (30, 30), "white")
    b = a.copy()
    b.putpixel((20, 20), (0, 0, 0))
    row = {"item_index": 0, "context": "mathematics", "bounds_px": [5, 5, 8, 8]}
    with pytest.raises(ValueError, match="outside recorded"):
        pixel_boundary_checks(a, b, [row], [row], {0})


def test_holdout_cannot_be_reused_with_different_receipt_or_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(render, "EVALUATION_LEDGER", tmp_path / "canonical-ledger")
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{"fixed":"candidate"}')
    first = tmp_path / "runs/first"
    marker = claim_holdout(freeze, first)
    before = marker.read_bytes()
    different_freeze = tmp_path / "new-candidate.json"
    different_freeze.write_text('{"fixed":"different candidate"}')
    with pytest.raises(FileExistsError):
        claim_holdout(different_freeze, tmp_path / "different-workspace/runs/retry")
    assert marker.read_bytes() == before


@pytest.mark.parametrize("changed", ["upstream_source", "mathtext_font"])
def test_freeze_rejects_changed_runtime(tmp_path: Path, monkeypatch, changed):
    monkeypatch.setattr(render, "source_paths", lambda: [])
    monkeypatch.setattr(render, "verify_run", lambda path: None)
    environment = {"upstream_source": "original", "mathtext_font": "original"}
    monkeypatch.setattr(render, "renderer_environment", lambda: environment.copy())
    baseline, legacy = tmp_path / "baseline", tmp_path / "legacy"
    for path in (baseline, legacy):
        path.mkdir()
        (path / "run.json").write_text("{}")
    protocol, registry = tmp_path / "protocol.md", tmp_path / "registry.json"
    protocol.write_text("frozen protocol")
    registry.write_text("{}")
    receipt = render.freeze_candidate(
        tmp_path / "freeze.json",
        protocol=protocol,
        registry=registry,
        baseline=baseline,
        legacy=legacy,
    )
    render.validate_freeze(receipt, baseline, legacy)
    environment[changed] = "modified"
    with pytest.raises(ValueError, match="environment, upstream code or MathText fonts"):
        render.validate_freeze(receipt, baseline, legacy)


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ({"realism": "combined"}, "baseline realism"),
        ({"profile_sha256": "b" * 64}, "Profile SHA-256"),
    ],
)
def test_manifest_cannot_silently_change_pinned_configuration(tmp_path, change, diagnostic):
    job = {
        "schema_version": 1,
        "generated_by": "test",
        "profile": "personal@v1",
        "problems": [
            {
                "number": "1",
                "statement": "A problem",
                "sources": [{"path": "source.txt"}],
                "solution": "answer.md",
            }
        ],
        **change,
    }
    manifest = tmp_path / "job.json"
    manifest.write_text(json.dumps(job))
    with pytest.raises(ValueError, match=diagnostic):
        render.manifest_cases(manifest, SimpleNamespace(manifest_sha256="a" * 64))


def test_missing_title_glyph_is_rejected_before_layout(tmp_path: Path, monkeypatch):
    for path in ("solutions/inputs/title", "reports", "logs"):
        (tmp_path / path).mkdir(parents=True)

    def unexpected_layout(*args, **kwargs):
        pytest.fail("Missing title glyph reached layout")

    monkeypatch.setattr(render, "layout", unexpected_layout)
    case = {
        "id": "title",
        "display_number": "题",
        "solution": "Answer.",
        "domain": "test",
        "split": "external",
        "source": {},
        "expected": {"outcome": "render"},
    }
    profile = SimpleNamespace(manifest=SimpleNamespace(coverage=list(range(32, 127))))
    result = render.render_case(case, tmp_path, RenderConfig(), profile, None, None, None)
    assert result["status"] == "rejected"
    assert "Missing glyph in problem number" in result["error"]


def test_white_ink_page_cannot_pass_pixel_invariance_alone():
    with pytest.raises(ValueError, match="Ink is not visible"):
        render.require_visible_ink(b"", Image.new("RGB", (20, 20), "white"), Background())
    image = Image.new("RGB", (20, 20), "white")
    image.putpixel((10, 10), (20, 20, 20))
    render.require_visible_ink(b"", image, Background())
