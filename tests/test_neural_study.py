"""Admission boundaries for optional learning; fixtures are not model-quality evidence."""

import json

import pytest
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from research.corpus.font_auxiliary import validate_auxiliary
from research.corpus.neural_study import PROTOCOL, admitted_inputs, validate_request


@pytest.fixture
def auxiliary(tmp_path):
    root = tmp_path / "auxiliary"
    root.mkdir()
    original = tmp_path / "source-profile.json"
    original.write_text("synthetic unit-test source")
    image = root / "glyph.png"
    Image.new("L", (64, 64), "white").save(image)
    row = dict(
        sample_id="fixture",
        label="1",
        codepoint=49,
        image=str(image),
        sha256=sha256(image),
        source_type="personal-font-derived",
        dataset_split="auxiliary-train-only",
        external_fallback=False,
        is_real_scan=False,
        personal_collected_outline=True,
    )
    atomic_json(root / "samples.json", [row])
    atomic_json(root / "metrics.json", {"fixture": True})
    atomic_json(
        root / "manifest.json",
        {
            "schema": "personal-font-auxiliary/1",
            "sample_count": 1,
            "personal_codepoints": [49],
            "source_files": {str(original): sha256(original)},
            "samples_sha256": sha256(root / "samples.json"),
            "metrics_sha256": sha256(root / "metrics.json"),
        },
    )
    return root


@pytest.mark.parametrize(
    "change",
    [
        {"external_fallback": True},
        {"is_real_scan": True},
        {"dataset_split": "holdout"},
        {"label": "l"},
        {"codepoint": 65, "label": "A"},
    ],
)
def test_auxiliary_never_becomes_real_or_holdout(auxiliary, change):
    rows, manifest = validate_auxiliary(auxiliary)
    rows[0].update(change)
    atomic_json(auxiliary / "samples.json", rows)
    manifest["samples_sha256"] = sha256(auxiliary / "samples.json")
    atomic_json(auxiliary / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Fallback"):
        validate_auxiliary(auxiliary)


def test_auxiliary_tampering_is_detected(auxiliary):
    rows, _ = validate_auxiliary(auxiliary)
    Image.new("L", (64, 64), "black").save(rows[0]["image"])
    with pytest.raises(ValueError, match="changed auxiliary"):
        validate_auxiliary(auxiliary)


def test_training_request_omits_holdout_and_rejects_later_injection(auxiliary, monkeypatch):
    from research.corpus import neural_study as study

    rows = [
        dict(sample_id=split, verified_text="1", review_status="verified", dataset_split=split)
        for split in ("train", "validation", "holdout")
    ]
    monkeypatch.setattr(
        study,
        "validate_dataset",
        lambda _: (
            rows,
            {
                "decision": "GO-limited-prototype",
                "eligible_labels": ["1"],
            },
        ),
    )
    real, fonts, labels = admitted_inputs(auxiliary, auxiliary)
    assert [r["dataset_split"] for r in real] == ["train", "validation"]
    request = dict(
        protocol=PROTOCOL,
        stage="real-prior",
        dataset=str(auxiliary),
        auxiliary=str(auxiliary),
        real_rows=[*real, rows[-1]],
        font_rows=fonts,
        labels=labels,
    )
    with pytest.raises(ValueError, match="holdout"):
        validate_request(request)
    request["protocol"] = json.loads(json.dumps(PROTOCOL))
    request["protocol"]["corrected_beta"] = 0.9
    with pytest.raises(ValueError, match="frozen"):
        validate_request(request)
