"""Read-only protection index and empirical inventory of inherited personal assets."""

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store

from .experiment import create


def audit(run_id: str) -> dict:
    run = create("corpus-initial-audit", run_id=run_id, config={"no_mutation_of_originals": True})
    protected = {}
    for root in ["profiles", "samples", "out", "workspace/runs", "workspace/checkpoints"]:
        for p in Path(root).rglob("*"):
            if p.is_file() and not p.resolve().is_relative_to(run.path):
                protected[str(p)] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    atomic_json(run.path / "reports/preservation-index.json", protected)
    atomic_json(
        run.path / "reports/state.json",
        {
            "latest_sha256": sha256(Path("workspace/latest.json")),
            "active_profile": profile_store().resolve("personal").provenance(),
            "runtime_source": {
                str(p): sha256(p)
                for p in Path("src/hwsol").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            },
            "runtime_dependencies": {
                str(p): sha256(p) for p in map(Path, ["pyproject.toml", "uv.lock"])
            },
            "vscode": {str(p): sha256(p) for p in Path(".vscode").rglob("*") if p.is_file()},
            "git": {
                ref: subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
                for ref in [
                    "main",
                    "v0.6.0^{}",
                    "experiment/diffink-cuda-v07",
                    "experiment/english-evidence-v08",
                    "HEAD",
                ]
            },
        },
    )
    scans = defaultdict(list)
    for root in ["samples", "profiles/realism-studio/samples"]:
        for p in Path(root).rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"} and p.is_file():
                scans[sha256(p)].append(str(p))
    profiles = []
    for p in sorted(Path("profiles/realism-studio/profiles/personal").glob("*/*/profile.json")):
        manifest = json.loads(p.read_text())
        fontpath = p.parent / manifest["font"]
        font = TTFont(fontpath)
        cmap = font.getBestCmap()
        glyphs = font.getGlyphSet()
        shapes = {}
        for cp in manifest["personal_codepoints"]:
            pen = RecordingPen()
            glyphs[cmap[cp]].draw(pen)
            import hashlib

            shapes[str(cp)] = hashlib.sha256(repr(pen.value).encode()).hexdigest()
        profiles.append(
            {
                "version": manifest["version"],
                "status": p.parent.parent.name,
                "manifest_sha256": sha256(p),
                "font_sha256": sha256(fontpath),
                "personal_codepoints": manifest["personal_codepoints"],
                "listed_shape_count": sum(manifest["variants"].values()),
                "variant_counts": manifest["variants"],
                "primary_contour_sha256": shapes,
                "asset_roles": Counter(a["role"] for a in manifest["assets"]),
                "build": manifest["build"],
                "source_indices": manifest["samples"],
            }
        )
        font.close()
    atomic_json(
        run.path / "reports/inventory.json",
        {
            "original_unique_files": len(scans),
            "duplicate_copies": sum(len(v) - 1 for v in scans.values()),
            "scan_groups": dict(scans),
            "profiles": profiles,
            "new_historical_pdfs": list(map(str, Path("source_pdf").glob("*.pdf"))),
            "template_acceptance_is_not_verified_transcription": True,
            "neural_training_completed": False,
        },
    )
    return run.finish(
        "passed",
        result={
            "protected_files": len(protected),
            "original_unique_files": len(scans),
            "profiles": len(profiles),
            "default_changed": False,
        },
    )


if __name__ == "__main__":
    import sys

    print(json.dumps(audit(sys.argv[1]), indent=2))
