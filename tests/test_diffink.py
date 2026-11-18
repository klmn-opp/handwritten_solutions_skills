"""Contract tests only: these fixtures are not evidence of neural inference."""

import json

import pytest
from PIL import Image

from hwsol.lab.diffink import DiffInkAdapter, full_stroke_preview, validate_reference


def aligned_reference():
    points = [[i, i % 10, 1, 0, 0] for i in range(200)]
    ends = [50, 100, 150, 200]
    for end in ends:
        points[end - 1][2:] = [0, 0, 1]
    return {
        "text": "abcd",
        "origin": "recorded-pen",
        "points": points,
        "character_end_indices": ends,
        "personal_collected": True,
    }


def test_reference_origin_alignment_and_vocabulary():
    ref = aligned_reference()
    constraints = validate_reference(ref, "abdc", dict.fromkeys("abcd"))
    assert constraints["personal_reference_claimed"]
    assert not constraints["generated_ink_personally_collected"]
    ref["origin"] = "estimated-trajectory"
    assert (
        validate_reference(ref, "abcd", dict.fromkeys("abcd"))["reference_origin"] == ref["origin"]
    )
    ref["origin"] = "scan-contour"
    with pytest.raises(ValueError, match="scan contour"):
        validate_reference(ref, "abcd", dict.fromkeys("abcd"))


@pytest.mark.parametrize("text", ["ab v", "babc", "abcdc", "a=b", "$a$"])
def test_unsupported_text_is_not_silently_substituted(text):
    with pytest.raises(ValueError):
        validate_reference(aligned_reference(), text, dict.fromkeys("abcd"))


def test_corrupt_character_alignment_rejected():
    ref = aligned_reference()
    ref["character_end_indices"][1] = 101
    with pytest.raises(ValueError, match="pen states"):
        validate_reference(ref, "abcd", dict.fromkeys("abcd"))
    ref = aligned_reference()
    ref["points"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_reference(ref, "abcd", dict.fromkeys("abcd"))


def test_preview_keeps_final_character_and_long_segment(tmp_path):
    points = [[0, 0, 1, 0, 0], [1, 1, 0, 0, 1], [2, 0, 1, 0, 0], [30, 1, 0, 0, 1]]
    result = full_stroke_preview(points, tmp_path)
    assert result["visible_segments"] == 2
    assert result["character_end_events"] == 2
    assert result["long_segments_not_removed"] == 2
    with Image.open(tmp_path / "pages/01.png") as image:
        assert image.getpixel((image.width - 20, 20)) == (0, 0, 0)
    assert (tmp_path / "strokes.svg").read_text().count("M ") == 2


def test_failed_adapter_retains_run_without_advancing_latest(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "latest.json").write_text('{"run_id":"previous"}')
    adapter = DiffInkAdapter(tmp_path / "python", tmp_path, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="settings"):
        adapter.render("abcd", tmp_path / "ref.json", steps=0, workspace=workspace, run_id="bad")
    assert json.loads((workspace / "runs/bad/run.json").read_text())["status"] == "failed"
    assert json.loads((workspace / "latest.json").read_text())["run_id"] == "previous"
