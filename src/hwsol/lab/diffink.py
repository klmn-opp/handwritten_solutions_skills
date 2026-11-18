"""Opt-in subprocess adapter for the reproduced, vocabulary-limited online continuation model."""

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run


def validate_reference(reference: dict, text: str, vocabulary: dict) -> dict:
    if not text or any(c in text for c in "$\\\n="):
        raise ValueError(
            "Neural backend accepts prose only; mathematical structure stays on font-cpu"
        )
    missing = sorted(set(text) - set(vocabulary))
    if missing:
        raise ValueError("DiffInk vocabulary missing: " + ", ".join(repr(c) for c in missing))
    if reference.get("origin") not in {
        "recorded-pen",
        "estimated-trajectory",
        "dataset-recorded-online",
    }:
        raise ValueError(
            "An aligned online trajectory is required; a scan contour is not a pen trajectory"
        )
    source = reference.get("text", "")
    points = reference.get("points", [])
    ends = reference.get("character_end_indices", [])
    if not source or len(ends) != len(source):
        raise ValueError("Text and character boundaries must align exactly")
    if not 200 <= len(points) <= 2000:
        raise ValueError("Supported reference length is 200 to 2000 observed points")
    if any(
        len(p) != 5 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p)
        for p in points
    ):
        raise ValueError("Trajectory requires finite absolute x,y and three pen-state values")
    if any(any(v not in (0, 1) for v in p[2:]) or sum(p[2:]) != 1 for p in points):
        raise ValueError("Pen states must be one-hot continue/stroke-end/character-end")
    if any(type(v) is not int for v in ends) or not all(0 < a <= len(points) for a in ends):
        raise ValueError("Character boundaries must index the observed sequence")
    if any(b <= a for a, b in zip(ends, ends[1:], strict=False)):
        raise ValueError("Character boundaries must increase")
    if ends != [i + 1 for i, point in enumerate(points) if point[4] == 1]:
        raise ValueError("Character boundaries must match observed character-end pen states")
    if ends[-1] != len(points):
        raise ValueError("Final character boundary must include all observed points")
    prefix = max(1, round(len(source) * 0.3))
    if len(text) != len(source) or text[:prefix] != source[:prefix]:
        raise ValueError(
            "This reproduced adapter requires equal character count and the same 30% prefix"
        )
    return {
        "reference_origin": reference["origin"],
        "prefix_characters": prefix,
        "reference_points": len(points),
        "requested_text": text,
        "personal_reference_claimed": bool(reference.get("personal_collected", False)),
        "generated_ink_personally_collected": False,
    }


def full_stroke_preview(points, folder: Path) -> dict:
    """Preserve all generated connecting segments; upstream's preview trims the final character."""
    if any(len(p) != 5 or any(not math.isfinite(float(v)) for v in p) for p in points):
        raise ValueError("Invalid generated stroke data")
    segments = [(a[:2], b[:2]) for a, b in zip(points, points[1:], strict=False) if a[2] >= 0.5]
    if not segments:
        raise ValueError("Generated sequence has no visible strokes")
    xy = [p for segment in segments for p in segment]
    x0, x1 = min(p[0] for p in xy), max(p[0] for p in xy)
    y0, y1 = min(p[1] for p in xy), max(p[1] for p in xy)
    scale = 216 / max(y1 - y0, 1e-8)
    width = math.ceil((x1 - x0) * scale) + 40
    if width > 20000 or width < 40:
        raise ValueError("Generated geometry has extreme aspect ratio")
    image = Image.new("RGB", (width, 256), "white")
    draw = ImageDraw.Draw(image)
    commands = []
    long_segments = 0
    for a, b in segments:
        start = ((a[0] - x0) * scale + 20, (y1 - a[1]) * scale + 20)
        end = ((b[0] - x0) * scale + 20, (y1 - b[1]) * scale + 20)
        draw.line([start, end], fill="black", width=2)
        commands.append(f"M {start[0]:.5f} {start[1]:.5f} L {end[0]:.5f} {end[1]:.5f}")
        long_segments += math.dist(start, end) >= 200
    (folder / "pages").mkdir(parents=True, exist_ok=True)
    image.save(folder / "pages/01.png")
    image.save(folder / "homework_solutions.pdf")
    (folder / "strokes.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="256" '
        f'viewBox="0 0 {width} 256">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<path d="{" ".join(commands)}" stroke="black" fill="none" stroke-width="2"/></svg>'
    )
    return {
        "visible_segments": len(segments),
        "long_segments_not_removed": long_segments,
        "preview": "all connected segments, no last-character truncation",
        "point_count": len(points),
        "character_end_events": sum(p[4] >= 0.5 for p in points),
    }


@dataclass
class DiffInkAdapter:
    python: Path
    checkout: Path
    assets_root: Path
    research: Path

    def render(
        self,
        text: str,
        reference: Path,
        *,
        seed=4206,
        steps=20,
        cfg_scale=1.0,
        workspace=None,
        run_id=None,
    ) -> dict:
        run = Run.create(
            "diffink-adapter",
            workspace=workspace,
            run_id=run_id,
            seed=seed,
            config={
                "backend": "diffink-experimental",
                "text": text,
                "steps": steps,
                "cfg_scale": cfg_scale,
            },
        )
        try:
            if not 1 <= steps <= 100 or not 0 <= cfg_scale <= 3 or not 0 <= seed < 2**31:
                raise ValueError("Invalid bounded inference settings")
            manifest = json.loads((self.research / "official-assets.json").read_text())["assets"]
            assets = {
                key: {**value, "path": str((self.assets_root / value["relative_path"]).resolve())}
                for key, value in manifest.items()
            }
            source = json.loads((self.research / "upstream-files.json").read_text())
            record = run.record()
            record["checkpoints"] = {k: v for k, v in assets.items() if k in {"vae", "dit"}}
            record["config"].update(
                upstream_commit=source["commit"],
                worker_sha256=sha256(self.research / "reproduce.py"),
                dependency_lock_sha256=sha256(self.research / "requirements-inference.lock"),
                asset_verification="pending isolated worker",
            )
            atomic_json(run.path / "run.json", record)
            vocabulary = json.loads(Path(assets["vocabulary"]["path"]).read_text())
            data = json.loads(reference.read_text())
            constraints = validate_reference(data, text, vocabulary)
            (run.path / "solutions/style-reference.json").write_bytes(reference.read_bytes())
            shutil.copyfile(
                self.research / "requirements-inference.lock",
                run.path / "reports/dependencies.lock",
            )
            request = {
                "run": str(run.path),
                "checkout": str(self.checkout.resolve()),
                "assets": assets,
                "seed": seed,
                "sampling_steps": steps,
                "cfg_scale": cfg_scale,
                "text": text,
                "reference": str(run.path / "solutions/style-reference.json"),
                "reference_sha256": sha256(reference),
                "upstream_source_sha256": source["files"],
                "dependency_lock": str(run.path / "reports/dependencies.lock"),
            }
            atomic_json(run.path / "reports/request.json", request)
            record = run.record()
            record["config"]["constraints"] = constraints
            atomic_json(run.path / "run.json", record)
            with (
                (run.path / "logs/stdout.txt").open("w") as stdout,
                (run.path / "logs/stderr.txt").open("w") as stderr,
            ):
                child = subprocess.run(
                    [
                        str(self.python.absolute()),
                        str((self.research / "reproduce.py").resolve()),
                        "--request",
                        str(run.path / "reports/request.json"),
                    ],
                    stdout=stdout,
                    stderr=stderr,
                    timeout=600,
                    check=False,
                )
            report_path = run.path / "reports/inference.json"
            if not report_path.is_file():
                raise ValueError(f"Inference worker exited {child.returncode}; see logs/stderr.txt")
            report = json.loads(report_path.read_text())
            if child.returncode or not report["model_executed"]:
                raise ValueError(report.get("error", "No genuine inference output"))
            import numpy as np

            points = np.load(
                run.path / "solutions/generated-strokes.npy", allow_pickle=False
            ).tolist()
            # Keep the official trimmed preview and expose the complete sequence separately.
            shutil.copyfile(
                run.path / "rendered/pages/01.png", run.path / "comparisons/official-preview.png"
            )
            preview = full_stroke_preview(points, run.path / "rendered")
            atomic_json(
                run.path / "reports/provenance.json",
                {
                    "source_type": "model-generated",
                    "personally_collected": False,
                    "style_reference_sha256": sha256(reference),
                    "style_reference_origin": data["origin"],
                    "checkpoint_sha256": {
                        k: v["sha256"] for k, v in assets.items() if k in {"vae", "dit"}
                    },
                    "requested_text": text,
                    "transcription": None,
                    "content_validated": False,
                    "profile": data.get("profile"),
                    "profile_sha256": data.get("profile_sha256"),
                    **preview,
                },
            )
            record = run.record()
            record["hardware"]["cuda"] = report["hardware"]
            record["config"]["asset_verification"] = "passed by isolated worker"
            atomic_json(run.path / "run.json", record)
            return run.finish(
                "held",
                result={
                    "actual_model_inference": "passed",
                    "content_and_style": "unvalidated",
                    "inference_seconds": report["inference_seconds"],
                    "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"],
                    "pdf": str(run.path / "rendered/homework_solutions.pdf"),
                },
            )
        except BaseException as exc:
            run.fail(exc)
            raise
