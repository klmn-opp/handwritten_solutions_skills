"""Exercise real scanned-image/PDF onboarding without using private handwriting."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from hwsol.cli import app
from hwsol.config import test_font as bundled_font
from hwsol.profiles.io import sha256
from hwsol.profiles.schema import ProfileError
from hwsol.profiles.store import ProfileStore
from hwsol.training.onboarding import build_from_scans
from hwsol.training.samples import list_samples
from hwsol.training.studio import publish
from hwsol.workspace import verify_run


@pytest.fixture(scope="module")
def scans(tmp_path_factory):
    from handnote.symbols import build_template, simulate_scan

    root = tmp_path_factory.mktemp("onboarding-fixture")
    pages, spec = build_template(chars="xA")
    spec_path = root / "template.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    image = root / "scan.jpg"
    simulate_scan(pages, spec, bundled_font(), [image], seed=42)
    pdf = root / "scan.pdf"
    with Image.open(image) as raster:
        raster.save(pdf, resolution=300)
    return spec_path, image, pdf


@pytest.mark.parametrize("scan_index", [1, 2], ids=["jpeg", "pdf"])
def test_reviewed_scans_build_preview_and_publish_locally(tmp_path, scans, scan_index):
    store = ProfileStore(tmp_path / "profiles")
    store.activate("bundled-default@v1")
    previous = store.resolve().manifest_sha256
    source, template = scans[scan_index], scans[0]
    source_hash = sha256(source)
    result = build_from_scans(store, "mine", [source], template, labels_reviewed=True)
    assert result["status"] == "held"
    assert result["engineering_evaluation"] == "passed"
    assert result["manual_style_review"] == "pending"
    assert result["candidate"] == "mine@v1"
    assert result["requested_without_personal_ink"] == ""
    assert set(result["personal_characters"]) >= {"A", "x"}
    assert not result["published"] and not result["activated"]
    assert store.resolve().manifest_sha256 == previous
    run = Path(result["run"])
    review = json.loads((run / "reports/label-review.json").read_text())
    assert review["originals"][0]["sha256"] == source_hash
    assert (run / "inputs" / review["originals"][0]["path"]).read_bytes() == source.read_bytes()
    assert review["template_sha256"] == sha256(template)
    assert Path(result["preview_pdf"]).is_file()
    assert list(Path(result["preview_pages"]).glob("*.png"))
    assert verify_run(run)["status"] == "passed"
    assert not (tmp_path / "workspace/latest.json").exists()
    assert sha256(source) == source_hash
    publish(store, result["candidate"], activate=True)
    assert store.resolve().manifest.ref == "mine@v1"
    assert verify_run(run)["status"] == "passed"  # publication leaves the receipt intact


def test_label_review_required_before_import(tmp_path, scans, monkeypatch):
    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "profiles"))
    result = CliRunner().invoke(
        app,
        [
            "profile",
            "build",
            "--profile",
            "mine",
            "--source",
            str(scans[1]),
            "--template",
            str(scans[0]),
        ],
    )
    assert result.exit_code == 2
    assert "Printed prompts alone" in result.output
    assert not list_samples(ProfileStore(tmp_path / "profiles"), "mine")


def test_missing_scan_pages_fail_without_candidate_and_retain_evidence(tmp_path, scans):
    template = tmp_path / "two-sets.json"
    spec = json.loads(scans[0].read_text())
    spec["sets"] = 2
    template.write_text(json.dumps(spec))
    store = ProfileStore(tmp_path / "profiles")
    with pytest.raises(ProfileError, match="Expected 2 scanned pages"):
        build_from_scans(store, "mine", [scans[1]], template, labels_reviewed=True)
    runs = list((tmp_path / "workspace/runs").iterdir())
    assert len(runs) == 1
    assert verify_run(runs[0])["run_status"] == "failed"
    assert (runs[0] / "inputs/scan-001.jpg").read_bytes() == scans[1].read_bytes()
    assert not list_samples(store, "mine")


def test_blank_ink_is_not_counted_as_personal_coverage(tmp_path):
    from handnote.symbols import build_template

    pages, spec = build_template(chars="xA")
    template = tmp_path / "blank.json"
    template.write_text(json.dumps(spec.to_dict()))
    source = tmp_path / "blank.png"
    pages[0].save(source)
    store = ProfileStore(tmp_path / "profiles")
    store.activate("bundled-default@v1")
    before = store.resolve().manifest_sha256
    with pytest.raises(ProfileError, match="No usable ink"):
        build_from_scans(store, "mine", [source], template, labels_reviewed=True)
    assert store.resolve().manifest_sha256 == before
    assert not store.version_names("mine", "candidates")
    run = next((tmp_path / "workspace/runs").iterdir())
    assert verify_run(run)["run_status"] == "failed"


def test_unfilled_cell_stays_explicit_fallback(tmp_path):
    from handnote.symbols import build_template, simulate_scan

    pages, spec = build_template(chars="xAB")
    template = tmp_path / "template.json"
    template.write_text(json.dumps(spec.to_dict()))
    scan = tmp_path / "partial.jpg"
    # The printed B remains unfilled; the actual observed synthetic ink is x and A.
    simulate_scan(pages, replace(spec, cells=spec.cells[:2]), bundled_font(), [scan], seed=42)
    store = ProfileStore(tmp_path / "profiles")
    result = build_from_scans(store, "mine", [scan], template, labels_reviewed=True)
    assert result["requested_without_personal_ink"] == "B"
    candidate = store.resolve(result["candidate"], candidate=True)
    assert ord("B") in candidate.manifest.coverage
    assert ord("B") not in candidate.manifest.personal_codepoints
    assert not result["published"] and not result["activated"]
