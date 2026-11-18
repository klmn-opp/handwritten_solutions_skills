"""Personal outline-derived auxiliary images; never real scan labels or holdout data."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256


def export_personal_font(profile, destination: Path) -> dict:
    from handnote.variants import handwritten_outline
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    from hwsol.profiles.metrics import measure_font

    destination.mkdir(parents=True, exist_ok=False)
    (destination / "images").mkdir()
    metrics = measure_font(profile.font, profile.manifest.personal_codepoints)
    atomic_json(destination / "metrics.json", metrics)
    em = 48 / metrics["references_em"]["digit_height"]
    records, excluded = [], []
    for row in metrics["glyphs"]:
        if row["excluded"]:
            excluded.append(row)
            continue
        vertices, codes = handwritten_outline(
            profile.font, row["codepoint"], row["slot"], em * row["scale"]
        )
        path = MplPath(vertices, codes)
        box = path.get_extents()
        vertices[:, 0] += 32 - (box.x0 + box.x1) / 2
        vertices[:, 1] += 56 - metrics["baseline_em"] * em * row["scale"]
        box = MplPath(vertices, codes).get_extents()
        # Fit only when needed, uniformly; never cut descenders or stretch one axis.
        fit = min(1.0, 56 / max(box.width, box.height))
        shift = [0.0, 0.0]
        if fit < 1 or box.y0 < 4 or box.y1 > 60:
            shift = [32 - (box.x0 + box.x1) * fit / 2, 32 - (box.y0 + box.y1) * fit / 2]
            vertices = vertices * fit + shift
        figure = Figure(figsize=(1, 1), dpi=256, facecolor="white")
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_axes([0, 0, 1, 1], xlim=(0, 64), ylim=(0, 64))
        axis.set_axis_off()
        axis.add_patch(PathPatch(MplPath(vertices, codes), facecolor="black", linewidth=0))
        canvas.draw()
        image = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("L")
        image = image.resize((64, 64), Image.Resampling.LANCZOS)
        name = f"u{row['codepoint']:04x}-v{row['slot']}.png"
        file = destination / "images" / name
        image.save(file)
        records.append(
            {
                "sample_id": f"font-{row['codepoint']}-{row['slot']}",
                "label": row["character"],
                "codepoint": row["codepoint"],
                "glyph_id": row["glyph_id"],
                "glyph_name": row["glyph_name"],
                "slot": row["slot"],
                "image": str(file.resolve()),
                "sha256": sha256(file),
                "source_type": "personal-font-derived",
                "label_source": "profile-cmap; not newly human-verified scan",
                "dataset_split": "auxiliary-train-only",
                "personal_collected_outline": True,
                "external_fallback": False,
                "is_real_scan": False,
                "metric_scale": row["scale"],
                "uniform_fit_scale": fit,
                "fit_translation": shift,
            }
        )
    atomic_json(destination / "samples.json", records)
    directory = profile.font.parent
    manifest = {
        "schema": "personal-font-auxiliary/1",
        "profile": f"{profile.manifest.profile}@{profile.manifest.version}",
        "source_files": {
            str((directory / name).resolve()): sha256(directory / name)
            for name in ("font.ttf", "profile.json", "sources.json")
        },
        "personal_codepoints": list(profile.manifest.personal_codepoints),
        "sample_count": len(records),
        "excluded": excluded,
        "samples_sha256": sha256(destination / "samples.json"),
        "metrics_sha256": sha256(destination / "metrics.json"),
        "independent_real_sample_count": 0,
        "scope": "auxiliary supervision only; no label admission or evaluation claims",
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def validate_auxiliary(directory: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != "personal-font-auxiliary/1":
        raise ValueError("Unknown auxiliary dataset schema")
    for key, name in [("samples_sha256", "samples.json"), ("metrics_sha256", "metrics.json")]:
        if sha256(directory / name) != manifest[key]:
            raise ValueError("Auxiliary manifest integrity mismatch")
    for path, digest in manifest["source_files"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source Profile changed")
    rows = json.loads((directory / "samples.json").read_text())
    if len(rows) != manifest["sample_count"]:
        raise ValueError("Auxiliary count mismatch")
    seen = set()
    for row in rows:
        if (
            row["source_type"] != "personal-font-derived"
            or row["dataset_split"] != "auxiliary-train-only"
            or row["external_fallback"]
            or row["is_real_scan"]
            or not row["personal_collected_outline"]
            or row["codepoint"] not in manifest["personal_codepoints"]
            or row["label"] != chr(row["codepoint"])
        ):
            raise ValueError("Fallback, nonpersonal or evaluation sample in auxiliary data")
        if row["sample_id"] in seen or sha256(Path(row["image"])) != row["sha256"]:
            raise ValueError("Duplicate or changed auxiliary image")
        seen.add(row["sample_id"])
    return rows, manifest
