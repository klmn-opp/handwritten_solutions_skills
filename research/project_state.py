"""Freeze the actual production and research lineage without changing production."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.jobs.runtime import verify_job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import load_snapshot
from hwsol.workspace import Run, profile_store, verify_run, workspace_root

RUNS = {
    "english_model": "20261120T022439649806Z-canonical-ink-component-refinement-fbea19c9",
    "english_layout": "20261120T040707343032Z-metric-neural-english-math-homework-88d48c44",
    "word_rhythm": "20261120T042107038172Z-measured-word-rhythm-mixed-homework-3be93094",
    "historical_math": "20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5",
    "session_math": "20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4",
    "research_best": "20261120T064909731073Z-v01026-session-math-polish-b9371278",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def verify_registry(root: Path) -> dict:
    results = {}
    for name in ("production", "research_best", "candidates"):
        record = read(root / "registry" / f"{name}.json")
        receipt = record["freeze_receipt"]
        if sha256(Path(receipt["path"])) != receipt["sha256"]:
            raise ValueError(f"Registry receipt changed: {name}")
        frozen = Path(receipt["path"]).parent
        verify_run(frozen)
        if record["payload"] != read(frozen / "reports" / f"{name}.json"):
            raise ValueError(f"Registry payload changed: {name}")
        protected = read(frozen / "reports/protected-assets.json")
        for path, expected in protected.items():
            if sha256(Path(path)) != expected:
                raise ValueError(f"Protected asset changed: {path}")
        results[name] = {"verified": True, "receipt": receipt}
    return results


def freeze(root: Path) -> Path:
    if any(
        (root / "registry" / f"{n}.json").exists()
        for n in ("production", "research_best", "candidates")
    ):
        raise ValueError(
            "Registry already exists; verify it instead of replacing its frozen baseline"
        )
    run = Run.create(
        "m0-baseline-freeze",
        workspace=root,
        seed=1909,
        config={
            "purpose": "read-only lineage audit and independent exact D replay",
            "plan": file_ref(Path("docs/MASTER_EXECUTION_PLAN.md")),
            "source": file_ref(Path(__file__)),
            "new_neural_updates": 0,
            "session_C_used": False,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/project_state.py")
        atomic_json(
            run.path / "reports/git.json",
            {
                "head": git("rev-parse", "HEAD"),
                "branch": git("branch", "--show-current"),
                "status": git("status", "--porcelain"),
                "refs": git("show-ref", "--heads", "--tags"),
                "main": git("rev-parse", "main"),
                "release": git("rev-parse", "v0.6.0^{commit}"),
            },
        )
        paths = {k: root / "runs" / v for k, v in RUNS.items()}
        latest = read(root / "latest.json")
        formal = root / latest["run"]
        if sha256(formal / "run.json") != latest["run_sha256"]:
            raise ValueError("Production pointer does not match its run")
        checks = {key: verify_run(path) for key, path in paths.items()}
        checks["production"] = verify_run(formal)
        checks["production_pixels"] = verify_job(formal / "rendered", pixels=True)
        atomic_json(run.path / "reports/verified-runs.json", checks)
        best = paths["research_best"]
        protected = read(best / "reports/protected-assets.json")
        protected.update(
            {str((p / "run.json").resolve()): sha256(p / "run.json") for p in paths.values()}
        )
        protected.update({str((root / "latest.json").resolve()): sha256(root / "latest.json")})
        settings = profile_store(root).root / "settings.json"
        if settings.exists():
            protected[str(settings.resolve())] = sha256(settings)
        model = paths["english_model"]
        request = read(model / "reports/request.json")
        result = read(model / "reports/result.json")
        checkpoint = model / f"logs/checkpoint-{request['steps']:05d}.pt"
        if sha256(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("Current neural checkpoint identity mismatch")
        protected[str(checkpoint.resolve())] = sha256(checkpoint)
        with np.load(model / "reports/targets.npz", allow_pickle=False) as data:
            coverage = {
                "real_views": int(data["real"].sum()),
                "real_classes": len(np.unique(data["labels"][data["real"]])),
                "font_only_classes": len(np.unique(data["labels"][~data["real"]])),
            }
        profiles = {
            str(v): load_snapshot(best / f"solutions/profile-v{v}").provenance() for v in (1, 3)
        }
        current_default = profile_store(root).resolve(None).provenance()
        atomic_json(run.path / "reports/protected-assets.json", protected)
        command = [
            sys.executable,
            "-m",
            "research.corpus.math_session_completion",
            "--selection",
            str(paths["session_math"] / "solutions/session-selection.json"),
            "--previous",
            str(paths["historical_math"]),
            "--polish-baseline",
            str(paths["session_math"]),
        ]
        atomic_json(run.path / "logs/replay-command.json", command)
        with (run.path / "logs/replay.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        replay = Path((run.path / "logs/replay.log").read_text().splitlines()[0])
        verify_run(replay)
        comparisons = []
        for arm in ("C", "D"):
            for index in range(1, 8):
                rel = f"rendered/pages/{arm}/page-{index:02d}.png"
                with Image.open(best / rel) as old, Image.open(replay / rel) as new:
                    equal = (
                        old.mode == new.mode
                        and old.size == new.size
                        and old.tobytes() == new.tobytes()
                    )
                if not equal:
                    raise ValueError(f"Frozen page differs: {rel}")
                comparisons.append({"path": rel, "pixels_equal": True})
        old_provenance = read(best / "reports/glyph-provenance.json")
        new_provenance = read(replay / "reports/glyph-provenance.json")
        # Source crop paths are run-local; semantic, layout and ink comparison is
        # independently done by the established session replay and page equality.
        if len(old_provenance) != len(new_provenance):
            raise ValueError("Page provenance count changed")
        for path, expected in protected.items():
            if sha256(Path(path)) != expected:
                raise ValueError(f"Historical asset changed during replay: {path}")
        shutil.copyfile(
            replay / "rendered/D-homework.pdf", run.path / "rendered/homework_solutions.pdf"
        )
        for index in range(1, 8):
            shutil.copyfile(
                replay / f"rendered/pages/D/page-{index:02d}.png",
                run.path / f"rendered/pages/{index:02d}.png",
            )
        production = {
            "backend": "font-cpu",
            "version": "0.6.0",
            "main_commit": git("rev-parse", "main"),
            "release_commit": git("rev-parse", "v0.6.0^{commit}"),
            "default_profile": current_default,
            "latest": latest,
            "formal_job_profile": read(formal / "rendered/manifest.json")["profile"],
            "production_pointer": file_ref(root / "latest.json"),
        }
        research = {
            "candidate": "v0.10.26 D",
            "status": "held",
            "production_admitted": False,
            "lineage_runs": {k: file_ref(p / "run.json") for k, p in paths.items()},
            "english_model": {
                "architecture": request["architecture"],
                "checkpoint": file_ref(checkpoint),
                "parameters": result["parameters"],
                "lineage_updates": result["model_lineage_updates"],
                "data": {k: file_ref(Path(request[k]) / "run.json") for k in ("real", "font")},
                "coverage": coverage,
                "sampling": "same-class seen PCA centre + 0.15 class std noise",
                "contextual_word_generation": False,
            },
            "profiles": profiles,
            "configuration": read(best / "run.json")["config"]["render_config"],
            "english_seed": 2012,
            "neural_words_first_four_pages": 526,
            "remaining_three_pages_english": "font-cpu",
            "word_gap_observations": 477,
            "math_source_manifests": [
                file_ref(paths["historical_math"] / "solutions/historical-glyphs/sources.json"),
                file_ref(best / "solutions/session-glyphs/sources.json"),
            ],
            "math_coverage_scope": "seven archived pages only; explicit future fallback required",
            "replay": file_ref(replay / "run.json"),
            "pdf": file_ref(run.path / "rendered/homework_solutions.pdf"),
            "new_neural_updates": 0,
        }
        candidates = {
            "scope": (
                "current frozen baseline and historical author-preferred comparison; "
                "not all archived experiments"
            ),
            "entries": [
                {
                    "id": "v01026-D",
                    "status": "held",
                    "role": "current_research_baseline",
                    "run": file_ref(best / "run.json"),
                },
                {
                    "id": "v01024-B",
                    "status": "held",
                    "author_feedback": "明显B的更好",
                    "scope": "v24 A/B only",
                    "run": file_ref(paths["historical_math"] / "run.json"),
                },
            ],
        }
        for name, payload in (
            ("production", production),
            ("research_best", research),
            ("candidates", candidates),
        ):
            atomic_json(run.path / "reports" / f"{name}.json", payload)
        atomic_json(run.path / "reports/pixel-replay.json", comparisons)
        run.finish(
            "passed",
            result={
                "purpose": "engineering baseline audit, not new style acceptance",
                "replay": str(replay),
                "exact_pages": len(comparisons),
                "protected_assets": len(protected),
                "checkpoint_verified": True,
                "source_coverage": coverage,
                "torch_loaded": "torch" in sys.modules,
                "production_unchanged": True,
            },
        )
        receipt = file_ref(run.path / "run.json")
        for name in ("production", "research_best", "candidates"):
            atomic_json(
                root / "registry" / f"{name}.json",
                {
                    "schema_version": 1,
                    "freeze_receipt": receipt,
                    "payload": read(run.path / "reports" / f"{name}.json"),
                },
            )
        verify_registry(root)
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("freeze", "verify"))
    cli.add_argument("--workspace", type=Path)
    args = cli.parse_args()
    root = workspace_root(args.workspace)
    if args.command == "freeze":
        freeze(root)
    else:
        print(json.dumps(verify_registry(root), indent=2))
