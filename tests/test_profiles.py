"""Phase 1 D/E/I/J: real upstream builds, lifecycle boundaries, portable CPU rendering."""

import json
import subprocess
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hwsol.adapters.handnote import HandnoteAdapter
from hwsol.cli import app
from hwsol.config import RenderConfig
from hwsol.config import test_font as bundled_font
from hwsol.probe import run_probe
from hwsol.profiles.fonts import outline_fingerprints
from hwsol.profiles.io import sha256
from hwsol.profiles.schema import ProfileError
from hwsol.profiles.store import ProfileStore
from hwsol.training.backend import HandnoteCPU
from hwsol.training.samples import import_sample, list_samples
from hwsol.training.studio import evaluate, publish, template, train


@pytest.fixture(scope="module")
def capture_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-template")
    result = template(root / "template.pdf", chars="xA")
    assert Path(result["pdf"]).is_file()
    assert Path(result["template"]).is_file()
    return root / "template.json"


def capture(root, round_no, template_path, store, *, quality=1):
    source = root / f"round-{round_no}.json"
    source.write_text(
        json.dumps(
            {
                "version": 1,
                "source": "synthetic-test-fixture",
                "cells": [
                    {
                        "index": 0,
                        "strokes": [
                            [[0.2, 0.2], [0.8, 0.8]],
                            [[0.2, 0.8], [0.8, 0.05 * round_no]],
                        ],
                    },
                ],
            }
        )
    )
    sample = import_sample(store, "personal", [source], template=template_path, quality=quality)
    assert sample.state == ("quarantined" if quality < 0.5 else "accepted")
    return sample, source


def release(store, candidate, activate=False):
    report = evaluate(store, candidate)
    assert report["status"] == "passed", report
    publish(store, candidate, activate=activate)


def render(root, store, ref=None, seed=42):
    selected = store.resolve(ref)
    root.mkdir()
    source = root / "input.md"
    source.write_text("xxxxxxxxxxxx $x^2=4,\\quad x=2$", encoding="utf-8")
    return run_probe(
        source,
        root / "result",
        selected.font,
        RenderConfig(seed=seed),
        HandnoteAdapter(real_variants=bool(selected.manifest.variants)),
        profile=selected.provenance(),
    )


def test_two_rounds_history_variants_rollback_and_portability(tmp_path, capture_template):
    store = ProfileStore(tmp_path / "data")
    sample1, source1 = capture(tmp_path, 1, capture_template, store)
    first = train(store, "personal")["candidate"]
    assert first == "personal@v1"
    with pytest.raises(ProfileError, match="Published Profile not found"):
        store.resolve(first)
    with pytest.raises(ProfileError, match="Evaluate"):
        publish(store, first)
    release(store, first, activate=True)
    loaded1 = store.resolve()
    first_hash = sha256(loaded1.font)
    old_shapes = outline_fingerprints(loaded1.font, ord("x"))
    image1 = render(tmp_path / "first", store)
    default = render(tmp_path / "default", store, "bundled-default@v1")
    assert image1["artifacts"]["pages/01.png"] != default["artifacts"]["pages/01.png"]

    sample2, _ = capture(tmp_path, 2, capture_template, store, quality=0.8)
    second = train(store, "personal")["candidate"]
    new = store.resolve(second, candidate=True)
    new_shapes = outline_fingerprints(new.font, ord("x"))
    assert len(new_shapes) == 2 and len(set(new_shapes)) == 2
    assert new_shapes[:1] == old_shapes
    assert new.manifest.parent == first
    assert new.manifest.variants[str(ord("x"))] == 2
    assert {s.id for s in new.manifest.samples} == {sample1.id, sample2.id}
    assert store.resolve().manifest.ref == first
    release(store, second)
    assert store.resolve().manifest.ref == first  # publication is not activation
    store.activate(second)
    image2 = render(tmp_path / "second", store)
    assert image2["artifacts"]["pages/01.png"] != image1["artifacts"]["pages/01.png"]
    assert store.compare(first, second)["removed_codepoints"] == []
    assert sha256(store.resolve(first).font) == first_hash
    originals = {s.id: (p, s) for p, s in list_samples(store, "personal")}
    p, s = originals[sample1.id]
    assert (p / s.originals[0]).read_bytes() == source1.read_bytes()
    store.activate(first)
    assert render(tmp_path / "rollback", store)["artifacts"] == image1["artifacts"]

    archive = tmp_path / "portable.zip"
    store.export(second, archive)
    with zipfile.ZipFile(archive) as z:
        assert not any("original-" in n or "sample.json" in n for n in z.namelist())
    other = ProfileStore(tmp_path / "another-computer")
    other.import_archive(archive)
    other.activate(second)
    assert not (other.root / "samples").exists()
    assert render(tmp_path / "portable", other)["artifacts"] == image2["artifacts"]
    with pytest.raises(ProfileError, match="already exists"):
        other.import_archive(archive)


def test_quality_quarantine_and_failed_build_preserve_current(tmp_path, capture_template):
    store = ProfileStore(tmp_path / "data")
    capture(tmp_path, 1, capture_template, store)
    release(store, train(store, "personal")["candidate"], activate=True)
    before = store.resolve()
    low, _ = capture(tmp_path, 3, capture_template, store, quality=0.1)
    new = store.resolve(train(store, "personal")["candidate"], candidate=True)
    assert outline_fingerprints(new.font, 120) == outline_fingerprints(before.font, 120)
    assert next(s for s in new.manifest.samples if s.id == low.id).included is False

    class BrokenBackend:
        def build(self, *args):
            raise RuntimeError("Simulated tracing failure")

    with pytest.raises(ProfileError, match="published versions unchanged"):
        train(store, "personal", BrokenBackend())
    assert store.resolve().manifest_sha256 == before.manifest_sha256
    assert store.version_names("personal", "candidates") == ["v2"]
    reports = list((store.root / "training-runs/personal").glob("r-*/run.json"))
    assert any(json.loads(p.read_text())["status"] == "failed" for p in reports)


def test_published_render_during_candidate_build(tmp_path, capture_template):
    store = ProfileStore(tmp_path / "data")
    capture(tmp_path, 1, capture_template, store)
    release(store, train(store, "personal")["candidate"], activate=True)
    initial = render(tmp_path / "initial", store)
    entered, resume = threading.Event(), threading.Event()

    class PausedBackend(HandnoteCPU):
        def build(self, *args):
            entered.set()
            if not resume.wait(timeout=20):
                raise RuntimeError("Test did not release builder")
            return super().build(*args)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(train, store, "personal", PausedBackend())
        try:
            assert entered.wait(timeout=10)
            assert render(tmp_path / "concurrent", store)["artifacts"] == initial["artifacts"]
            assert store.version_names("personal", "candidates") == []
            with pytest.raises(ProfileError, match="Another writer"):
                train(store, "personal")
        finally:
            resume.set()
        assert future.result(timeout=10)["candidate"] == "personal@v2"
    assert store.resolve().manifest.ref == "personal@v1"


def test_runtime_without_studio_imports(tmp_path):
    # The import hook makes absence of Studio and neural packages an executable boundary.
    code = """
import importlib.abc, json, sys
from pathlib import Path
class BlockStudio(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(("hwsol.training", "torch", "tensorflow", "onnxruntime")):
            raise ImportError("Training dependency forbidden: " + fullname)
sys.meta_path.insert(0, BlockStudio())
from typer.testing import CliRunner
from hwsol.cli import app
root = Path(sys.argv[1]); src = root / "input.md"; src.write_text("Runtime $x=1$")
result = CliRunner().invoke(app, ["render-probe", "--input", str(src), "--output", str(root/"out")])
assert result.exit_code == 0, result.output
job = Path(sys.argv[2]) / "examples/phase2/euler/job.json"
result = CliRunner().invoke(app, ["render", "--manifest", str(job), "--output", str(root/"job"),
                                 "--realism", "combined"])
assert result.exit_code == 0, (result.output, result.exception)
assert not any(n.startswith("hwsol.training") for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(Path(__file__).resolve().parents[1])],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_scan_and_font_import_use_real_backends(tmp_path):
    from handnote.symbols import build_template, simulate_scan

    pages, spec = build_template(chars="xA")
    paths = [tmp_path / f"scan-{n}.jpg" for n in range(len(pages))]
    simulate_scan(pages, spec, bundled_font(), paths, seed=42)
    spec_path = tmp_path / "scan.json"
    spec_path.write_text(json.dumps(spec.to_dict()))
    store = ProfileStore(tmp_path / "data")
    sample = import_sample(store, "scanned", paths, template=spec_path)
    assert sample.state == "accepted" and not sample.has_recorded_strokes
    candidate = train(store, "scanned")["candidate"]
    loaded = store.resolve(candidate, candidate=True)
    assert set(loaded.manifest.personal_codepoints) >= {65, 120}
    sample = import_sample(store, "font-copy", [loaded.font])
    assert sample.kind == "font"
    candidate2 = train(store, "font-copy")["candidate"]
    assert store.resolve(candidate2, candidate=True).manifest.coverage == loaded.manifest.coverage


def test_unstructured_and_malformed_samples_are_preserved(tmp_path, capture_template):
    store = ProfileStore(tmp_path / "data")
    for filename, content, expected in [
        ("note.pdf", b"raw unstructured PDF bytes", "archived"),
        ("broken.ttf", b"invalid font bytes", "quarantined"),
        ("broken.json", b'{"cells": []}', "quarantined"),
    ]:
        path = tmp_path / filename
        path.write_bytes(content)
        sample = import_sample(
            store,
            "incomplete",
            [path],
            template=capture_template if filename.endswith("json") else None,
        )
        assert sample.state == expected and not sample.has_recorded_strokes
        folder = store.root / "samples/incomplete" / sample.id
        assert (folder / sample.originals[0]).read_bytes() == content
    with pytest.raises(ProfileError, match="No accepted samples"):
        train(store, "incomplete")


def test_failed_evaluation_and_tampering_cannot_publish(tmp_path, capture_template, monkeypatch):
    store = ProfileStore(tmp_path / "data")
    capture(tmp_path, 1, capture_template, store)
    spec = train(store, "personal")["candidate"]
    with monkeypatch.context() as context:

        def failed_render(*args, **kwargs):
            raise RuntimeError("Simulated evaluation rendering failure")

        context.setattr("hwsol.training.studio.run_probe", failed_render)
        failed = evaluate(store, spec)
        assert failed["status"] == "failed" and "rendering failure" in failed["error"]["message"]
        with pytest.raises(ProfileError, match="Evaluation failed"):
            publish(store, spec)
    report = evaluate(store, spec)
    assert report["status"] == "passed"
    assessment = store.profile_dir("personal") / "evaluations/v1.json"
    missing = json.loads(assessment.read_text())
    missing["rendering"]["artifacts"] = {}
    assessment.write_text(json.dumps(missing))
    with pytest.raises(ProfileError, match="inventory"):
        publish(store, spec)
    assessment.write_text(json.dumps(report))
    preview = Path(report["run"]) / "preview/pages/01.png"
    preview.write_bytes(b"corrupted preview")
    with pytest.raises(ProfileError, match="preview integrity"):
        publish(store, spec)
    assert store.version_names("personal") == []
    font = store.resolve(spec, candidate=True).font
    font.write_bytes(font.read_bytes() + b"changed")
    with pytest.raises(ProfileError, match="integrity"):
        publish(store, spec)


@pytest.mark.parametrize("entry", ["../escape", "/absolute", "C:/drive", "a\\b", "link"])
def test_portable_archive_rejects_unsafe_entries(tmp_path, entry):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo(entry)
        if entry == "link":
            info.create_system = 3
            info.external_attr = 0o120777 << 16
        z.writestr(info, "outside")
    with pytest.raises(ProfileError):
        ProfileStore(tmp_path / "data").import_archive(archive)
    assert not (tmp_path / "escape").exists()


def test_bundled_read_only_cli_selection_and_machine_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "data"))
    store = ProfileStore()
    assert store.resolve().manifest.source_kind == "test-font"
    with pytest.raises(ProfileError, match="read-only"):
        store.create("bundled-default")
    runner = CliRunner()
    result = runner.invoke(
        app, ["profile", "template", "--output", str(tmp_path / "form.pdf"), "--chars", "xA"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "form.json").is_file()
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    result = runner.invoke(app, ["profiles", "import", "--source", str(bad)])
    assert result.exit_code == 2 and json.loads(result.stderr)["status"] == "failed"
    result = runner.invoke(app, ["profiles", "rollback", "bundled-default"])
    assert result.exit_code == 2 and "requires profile@version" in result.stderr


def test_more_than_eight_rounds_retain_real_variants(tmp_path, capture_template):
    store = ProfileStore(tmp_path / "data")
    for n in range(1, 10):
        capture(tmp_path, n, capture_template, store)
    ref = train(store, "personal")["candidate"]
    shapes = outline_fingerprints(store.resolve(ref, candidate=True).font, 120)
    assert len(shapes) == len(set(shapes)) == 9
    release(store, ref, activate=True)
    first = render(tmp_path / "seed-first", store, seed=17)["artifacts"]
    assert render(tmp_path / "seed-repeat", store, seed=17)["artifacts"] == first
    changed = render(tmp_path / "seed-different", store, seed=18)["artifacts"]
    assert first["pages/01.png"] != changed["pages/01.png"]


def test_cff_otf_standalone_and_explicit_merge_limit(tmp_path):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.t2CharStringPen import T2CharStringPen

    # Synthetic CFF boxes exercise the file format, not handwriting quality.
    builder = FontBuilder(1000, isTTF=False)
    cmap = {cp: f"u{cp:04X}" for cp in range(32, 127)}
    glyphs = [".notdef", *cmap.values()]
    builder.setupGlyphOrder(glyphs)
    builder.setupCharacterMap(cmap)
    strings = {}
    for glyph in glyphs:
        pen = T2CharStringPen(600, None)
        if glyph != cmap[32]:
            pen.moveTo((50, 0))
            pen.lineTo((500, 0))
            pen.lineTo((500, 700))
            pen.lineTo((50, 700))
            pen.closePath()
        strings[glyph] = pen.getCharString()
    builder.setupCFF("SyntheticCFF", {"FullName": "Synthetic CFF"}, strings, {})
    builder.setupHorizontalMetrics({glyph: (600, 50) for glyph in glyphs})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "Synthetic CFF", "styleName": "Regular"})
    builder.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    builder.setupPost()
    source = tmp_path / "fixture.otf"
    builder.save(source)
    store = ProfileStore(tmp_path / "data")
    import_sample(store, "cff", [source])
    candidate = train(store, "cff")["candidate"]
    assert store.resolve(candidate, candidate=True).font.suffix == ".otf"
    release(store, candidate, activate=True)
    assert render(tmp_path / "otf-render", store)["status"] == "passed"
    import_sample(store, "cff", [source])
    with pytest.raises(ProfileError, match="CFF OTF can be used alone"):
        train(store, "cff")
    assert store.resolve().manifest.ref == candidate


def test_personal_math_cache_and_variants_in_production_job(tmp_path, capture_template):
    from hwsol.jobs.layout import render_one
    from hwsol.jobs.schema import Background

    store = ProfileStore(tmp_path / "data")
    for n in (1, 2):
        capture(tmp_path, n, capture_template, store)
    ref = train(store, "personal")["candidate"]
    release(store, ref)
    source = r"xx $x+x$ and $x+x$, with $\sigma$."
    for profile in (store.resolve("bundled-default@v1"), store.resolve(ref)):
        _, report = render_one(source, "1", profile, RenderConfig(), Background(), None, "combined")
    uses = [g for g in report["glyph_sources"] if g["codepoint"] == "U+0078"]
    assert {g["source"] for g in uses} == {"personal"}
    assert {g["variant"] for g in uses if g["route"] == "math"} == {0, 1}
    assert sum(g["count"] for g in uses) == 6
    assert report["math_fallback_glyphs"] == [{"codepoint": "U+03C3", "font": "STIXGeneral.ttf"}]
