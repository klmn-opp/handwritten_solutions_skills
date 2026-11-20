"""Fixed English prompts, blinded image import and character-level evaluation."""

import json
import random
import tempfile
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.schema import Digest

MODELS = {
    "diffusionpen": ("koninik/DiffusionPen", "059e2f2bcb7372231038719e955597917c1a8783", "word"),
    "diffbrush": ("dailenson/DiffBrush", "da9addc1140bdfec463b2d41a974a0a392f80798", "line"),
    "one-dm": ("dailenson/One-DM", "dde2205a70a2c70d1786503d198a795358c80ee4", "word"),
}
WORDS = (
    "minimum",
    "momentum",
    "parallel",
    "Median",
    "gradient",
    "vector",
    "sample",
    "velocity",
    "limit",
    "matrix",
    "zero",
    "unit",
)
# Upstream DiffBrush accepts lines of length 35 through 61.
LINES = (
    "The median is the middle ordered value.",
    "Use the same units for each measurement.",
    "The two parallel lines have equal slope.",
    "Record the velocity at the final point.",
    "The sample mean changes with each trial.",
    "Check the sign before taking the limit.",
    "The gradient points uphill at this point.",
    "Read the minimum value from the table.",
)


def catalog():
    entries = {
        key: {
            "repository": f"https://github.com/{repo}",
            "revision": rev,
            "scope": scope,
            "code_license": "MIT",
            "runtime_enabled": False,
        }
        for key, (repo, rev, scope) in MODELS.items()
    }
    entries["diffink"] = {
        "repository": "https://github.com/awei669/DiffInk",
        "revision": "65055545fbef5e22fbee0811a1ef3855ca242c22",
        "scope": "aligned online continuation, vocabulary-limited",
        "code_license": "MIT",
        "weights_license": "CASIA research terms",
        "runtime_enabled": False,
        "command": "hwsol lab diffink --help",
    }
    return entries


def prompts(model):
    if model not in MODELS:
        raise ValueError(f"Unknown research model: {model}")
    return LINES if MODELS[model][2] == "line" else WORDS


def prepare(model: str, output: Path):
    words = prompts(model)
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompts.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    atomic_json(
        output / "run.json",
        {
            "protocol": 1,
            "model": model,
            "revision": MODELS[model][1],
            "seed": None,
            "device": "",
            "checkpoint_sha256": None,
            "style_reference_sha256": None,
            "style_reference_file": None,
            "rights_evidence_file": None,
        },
    )
    return {"status": "prepared", "samples": len(words), "model": model}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Trial(Strict):
    protocol: int
    model: str
    revision: str
    seed: int | None
    device: str
    checkpoint_sha256: Digest | None
    style_reference_sha256: Digest | None
    style_reference_file: str | None = None
    rights_evidence_file: str | None


class Reading(Strict):
    image_sha256: Digest
    transcription: str = Field(max_length=256)
    style: int = Field(ge=1, le=5)


class Review(Strict):
    reviewer: str = Field(min_length=1)
    readings: dict[str, Reading]


def trial(run):
    settings = Trial.model_validate_json((run / "run.json").read_bytes())
    expected = prompts(settings.model)
    if settings.protocol != 1 or settings.revision != MODELS[settings.model][1]:
        raise ValueError("Unknown protocol or upstream revision")
    if (run / "prompts.txt").read_text("utf-8").splitlines() != list(expected):
        raise ValueError("Fixed corpus was changed")
    return settings, expected


def image_for(source, prompt, index, model, writer, numbered):
    if numbered:
        return inside(source, f"{index:02d}.png")
    if model == "one-dm":
        return inside(source, f"{prompt}.png")
    if not writer:
        raise ValueError("Supply the exact upstream writer ID with --writer")
    if model == "diffusionpen":
        return inside(source, f"{prompt}_style_{writer}.png")
    prefix = f"{writer}-{prompt}_"
    matches = [p for p in source.iterdir() if p.name.startswith(prefix) and p.suffix == ".png"]
    if len(matches) != 1:
        raise ValueError(f"Expected one upstream PNG for prompt {index}; found {len(matches)}")
    return inside(source, matches[0].name)


def collect(run: Path, source: Path, writer: str | None, numbered: bool):
    settings, corpus = trial(run)
    dest = run / "batch"
    if dest.exists():
        raise ValueError("This run already contains a review batch")
    with tempfile.TemporaryDirectory(dir=run) as tmp:
        batch = Path(tmp) / "batch"
        blind = batch / "blind"
        blind.mkdir(parents=True)
        order = list(enumerate(corpus, 1))
        random.SystemRandom().shuffle(order)
        key, readings = {}, {}
        for serial, (index, prompt) in enumerate(order, 1):
            original = image_for(source, prompt, index, settings.model, writer, numbered)
            label = f"r{serial:02d}"
            target = blind / f"{label}.png"
            with Image.open(original) as image:
                if image.format != "PNG" or image.width * image.height > 8_000_000:
                    raise ValueError("Expected a PNG word/line crop of at most 8 megapixels")
                rgba = image.convert("RGBA")
                opaque = Image.new("RGB", rgba.size, "white")
                opaque.paste(rgba, mask=rgba.getchannel("A"))
                if all(hi - lo < 8 for lo, hi in opaque.getextrema()):
                    raise ValueError(f"Empty or unreadable crop for prompt {index}")
                # Re-encode without upstream metadata or filenames that disclose prompt text.
                opaque.save(target)
            readings[label] = {"image_sha256": sha256(target), "transcription": None, "style": None}
            key[label] = {
                "expected": prompt,
                "original_sha256": sha256(original),
                "image_sha256": readings[label]["image_sha256"],
            }
        atomic_json(blind / "review.json", {"reviewer": "", "readings": readings})
        atomic_json(batch / "key.json", key)
        batch.rename(dest)
    return {"status": "ready_for_blind_review", "samples": len(key)}


def edits(expected, observed):
    """Levenshtein insertions, deletions and substitutions, preserving case and spaces."""
    prev = list(range(len(observed) + 1))
    for i, char in enumerate(expected, 1):
        row = [i]
        for j, other in enumerate(observed, 1):
            row.append(min(prev[j] + 1, row[-1] + 1, prev[j - 1] + (char != other)))
        prev = row
    return prev[-1]


def evaluate(run: Path):
    settings, corpus = trial(run)
    key = json.loads((run / "batch/key.json").read_text("utf-8"))
    blind = run / "batch/blind"
    review = Review.model_validate_json((blind / "review.json").read_bytes())
    ids = {f"r{i:02d}" for i in range(1, len(corpus) + 1)}
    if (
        set(key) != ids
        or set(review.readings) != ids
        or sorted(entry["expected"] for entry in key.values()) != sorted(corpus)
    ):
        raise ValueError("Review must cover every fixed prompt exactly once")
    rows = []
    for label in sorted(ids):
        reading = review.readings[label]
        if not (
            sha256(inside(blind, f"{label}.png"))
            == reading.image_sha256
            == key[label]["image_sha256"]
        ):
            raise ValueError(f"Reviewed image changed: {label}")
        expected = key[label]["expected"]
        rows.append(
            {
                "id": label,
                "expected": expected,
                **reading.model_dump(),
                "original_sha256": key[label]["original_sha256"],
                "edits": edits(expected, reading.transcription),
            }
        )
    errors = sum(row["edits"] for row in rows)
    chars = sum(map(len, corpus))
    mean_style = sum(row["style"] for row in rows) / len(rows)
    style_ref = None
    if settings.style_reference_file:
        style_ref = sha256(inside(run, settings.style_reference_file))
        if style_ref != settings.style_reference_sha256:
            raise ValueError("Style reference differs from declared SHA-256")
    sampling = run / "raw/sampling.json"
    if sampling.exists():
        record = json.loads(sampling.read_bytes())
        if (
            record["status"] != "sampled"
            or record["revision"] != settings.revision
            or record["checkpoint_sha256"] != settings.checkpoint_sha256
        ):
            raise ValueError("Upstream sampling record does not match trial provenance")
    rights = None
    if settings.rights_evidence_file:
        file = inside(run, settings.rights_evidence_file)
        if not file.is_file() or not file.read_text("utf-8").strip():
            raise ValueError("Rights evidence needs a nonempty local text document")
        rights = sha256(file)
    gates = {
        "exact_text": errors == 0,
        "distinct_images": len({row["image_sha256"] for row in rows}) == len(rows),
        "style": mean_style >= 4 and min(row["style"] for row in rows) >= 3,
        "provenance": bool(
            settings.checkpoint_sha256
            and style_ref
            and settings.seed is not None
            and settings.device.strip()
            and rights
        ),
    }
    return {
        "status": "eligible_for_larger_trial" if all(gates.values()) else "held",
        "runtime_enabled": False,
        "gates": gates,
        "run": settings.model_dump(),
        "rights_evidence_sha256": rights,
        "reviewer": review.reviewer,
        "edits": errors,
        "characters": chars,
        "cer": errors / chars,
        "exact_samples": sum(r["edits"] == 0 for r in rows),
        "sample_count": len(rows),
        "mean_style": mean_style,
        "samples": rows,
    }
