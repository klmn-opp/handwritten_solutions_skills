"""Research protocol regression using identified synthetic crops. No inference is claimed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from hwsol.cli import app
from hwsol.lab import protocol
from hwsol.lab.protocol import LINES, WORDS
from hwsol.lab.sampling import sample
from hwsol.profiles.io import atomic_json, sha256


def cli(*args, code=0):
    result = CliRunner().invoke(app, list(map(str, args)))
    assert result.exit_code == code, (result.output, result.exception)
    return json.loads(result.stdout or result.stderr)


def batch(tmp_path, model):
    run, source = tmp_path / "trial", tmp_path / "writer"
    cli("lab", "prepare", "--model", model, "--output", run)
    corpus = (run / "prompts.txt").read_text("utf-8").splitlines()
    source.mkdir()
    for prompt in corpus:
        name = {
            "one-dm": f"{prompt}.png",
            "diffbrush": f"writer-{prompt}_0.png",
            "diffusionpen": f"{prompt}_style_writer.png",
        }[model]
        image = Image.new("L", (350, 30), "white")
        ImageDraw.Draw(image).text((4, 4), prompt, fill="black")
        image.save(source / name)
    cli("lab", "collect", "--run", run, "--source", source, "--writer", "writer")
    return run


@pytest.mark.parametrize("model", ["one-dm", "diffbrush", "diffusionpen"])
def test_native_output_import_and_blind_measurement(tmp_path, model):
    run = batch(tmp_path, model)
    blind = run / "batch/blind"
    key = json.loads((run / "batch/key.json").read_bytes())
    sheet = json.loads((blind / "review.json").read_bytes())
    assert all("expected" not in item for item in sheet["readings"].values())
    settings = json.loads((run / "run.json").read_bytes())
    sheet["reviewer"] = "synthetic regression"
    for key_id, item in sheet["readings"].items():
        item.update(transcription=key[key_id]["expected"], style=4)
    atomic_json(blind / "review.json", sheet)
    held = cli("lab", "evaluate", "--run", run, code=2)
    assert held["cer"] == 0 and held["status"] == "held" and not held["gates"]["provenance"]
    (run / "rights.txt").write_text("Synthetic test record", encoding="utf-8")
    shutil.copyfile(blind / "r01.png", run / "reference.png")
    settings.update(
        seed=42,
        device="fixture",
        checkpoint_sha256="0" * 64,
        style_reference_sha256=sha256(run / "reference.png"),
        style_reference_file="reference.png",
        rights_evidence_file="rights.txt",
    )
    atomic_json(run / "run.json", settings)
    record = cli("lab", "evaluate", "--run", run)
    assert record["status"] == "eligible_for_larger_trial" and not record["runtime_enabled"]
    assert record["sample_count"] == len(LINES if model == "diffbrush" else WORDS)
    row = sheet["readings"]["r01"]
    row.update(transcription=row["transcription"][:-1], style=1)
    atomic_json(blind / "review.json", sheet)
    held = cli("lab", "evaluate", "--run", run, code=2)
    assert held["edits"] == 1 and held["cer"] == 1 / held["characters"]
    assert not held["gates"]["style"]
    shutil.copyfile(blind / "r02.png", blind / "r01.png")
    assert "changed" in cli("lab", "evaluate", "--run", run, code=2)["error"]


def test_incomplete_batch_never_publishes(tmp_path):
    run = tmp_path / "trial"
    cli("lab", "prepare", "--model", "one-dm", "--output", run)
    cli("lab", "collect", "--run", run, "--numbered", "--source", tmp_path, code=2)
    assert not (run / "batch").exists()
    assert protocol.edits("abc", "") == 3
    assert protocol.edits("abc", "ABC") == 3
    assert protocol.edits("a b", "a  b") == 1


@pytest.mark.parametrize("model", ["diffbrush", "one-dm"])
def test_sampler_protocol_with_local_stand_in(tmp_path, monkeypatch, model):
    # Exercise pins, cwd, upstream argv, offline flags, hashes and failure reporting.
    # Never execute a model or assert its performance in this fixture.
    from hwsol.lab.sampling import ENTRIES

    script, config, corpus_location, flag = ENTRIES[model]
    checkout, assets, run = (tmp_path / p for p in ("checkout", "assets", "run"))
    checkout.mkdir()
    (checkout / config).parent.mkdir()
    (checkout / config).write_text("", encoding="utf-8")
    (checkout / script).write_text("", encoding="utf-8")
    for cmd in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-qm", "fixture"],
    ):
        subprocess.run(["git", *cmd], cwd=checkout, check=True)
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    repo, _, scope = protocol.MODELS[model]
    monkeypatch.setitem(protocol.MODELS, model, (repo, rev, scope))
    protocol.prepare(model, run)
    (assets / corpus_location).parent.mkdir(parents=True)
    shutil.copyfile(run / "prompts.txt", assets / corpus_location)
    (assets / "vae").mkdir()
    (assets / "vae/config.json").write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "fixture.weight"
    checkpoint.write_bytes(b"fake")
    launcher = Path(shutil.which("git"))
    args = (run, checkout, assets, launcher, checkpoint, assets, 1)
    prepared = sample(*args, True)
    assert flag in prepared["argv"] and prepared["checkpoint_sha256"] == sha256(checkpoint)
    assert not (run / "raw").exists()
    # A real subprocess fails. The raw run keeps its diagnostic and never enters review.
    with pytest.raises(ValueError, match="Upstream exited"):
        sample(*args, False)
    assert json.loads((run / "raw/sampling.json").read_bytes())["status"] == "failed"
    assert not (run / "batch").exists()
