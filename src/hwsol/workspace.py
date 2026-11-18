"""Immutable experiment receipts and a verified-deliverable pointer, independent of backends."""

import inspect
import json
import os
import platform
import re
import shutil
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from hwsol.identity import build_identity
from hwsol.profiles.io import atomic_json, inside, lock, sha256

DIRECTORIES = ("solutions", "rendered/pages", "reports", "comparisons", "logs")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}\Z")


def workspace_root(path: Path | None = None) -> Path:
    return (path or Path(os.environ.get("HWSOL_WORKSPACE", "workspace"))).expanduser().resolve()


def profile_store(workspace: Path | None = None):
    from hwsol.profiles.store import ProfileStore

    settings = workspace_root(workspace) / "workspace.json"
    if not os.environ.get("HWSOL_DATA_DIR") and settings.is_file():
        configured = json.loads(settings.read_text()).get("profile_store")
        if configured:
            return ProfileStore(Path(configured))
    return ProfileStore()


def hardware() -> dict:
    """No model imports and no device allocation; CUDA readiness is a separate lab check."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "cuda": "not_probed",
    }


def initialize(root: Path, profile_store: Path | None = None) -> dict:
    root = workspace_root(root)
    root.mkdir(parents=True, exist_ok=True)
    config = root / "workspace.json"
    if config.exists():
        raise ValueError("Workspace is already initialized; configuration is not overwritten")
    for area in ("runs", "archive", "datasets", "checkpoints", "environments", "sources"):
        (root / area).mkdir(exist_ok=True)
    record = {"schema_version": 1, "profile_store": str((profile_store or root / "data").resolve())}
    atomic_json(config, record)
    return record


class Run:
    def __init__(self, path: Path):
        self.path = path.resolve()

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        workspace: Path | None = None,
        run_id: str | None = None,
        config: dict | None = None,
        seed: int | None = None,
        profile: dict | None = None,
        checkpoints: dict | None = None,
    ):
        root = workspace_root(workspace)
        run_id = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{kind}-{uuid.uuid4().hex[:8]}"
        if not RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
            raise ValueError("Invalid immutable run ID")
        path = inside(root, f"runs/{run_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()  # Exclusive allocation: failed IDs cannot be reused either.
        for directory in DIRECTORIES:
            (path / directory).mkdir(parents=True, exist_ok=True)
        run = cls(path)
        atomic_json(
            path / "run.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "kind": kind,
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
                "build": build_identity(),
                "config": config or {},
                "seed": seed,
                "profile": profile,
                "checkpoints": checkpoints or {},
                "hardware": hardware(),
                "validation": {"content": "not_evaluated", "style": "not_evaluated"},
            },
        )
        return run

    def record(self) -> dict:
        return json.loads((self.path / "run.json").read_text("utf-8"))

    def finish(self, status: str, *, result: dict | None = None, deliverable: bool = False) -> dict:
        if status not in {"passed", "failed", "blocked", "prepared", "held"}:
            raise ValueError("Invalid terminal run status")
        with lock(self.path / ".finalize.lock"):
            record = self.record()
            if record["status"] != "running":
                raise ValueError("Finalized runs are immutable; allocate a new run ID")
            if deliverable:
                if status != "passed":
                    raise ValueError("Only validated successful deliveries may advance latest.json")
                from hwsol.jobs.runtime import verify_job

                verify_job(self.path / "rendered", pixels=True)
            atomic_json(self.path / "reports/result.json", result or {})
            inventory = {
                str(p.relative_to(self.path)): sha256(p)
                for p in sorted(self.path.rglob("*"))
                if p.is_file() and p.name not in {"run.json", ".finalize.lock"}
            }
            record.update(
                status=status,
                ended_at=datetime.now(UTC).isoformat(),
                result=result or {},
                artifacts=inventory,
                deliverable=deliverable,
            )
            atomic_json(self.path / "run.json", record)
            if deliverable:
                root = self.path.parent.parent
                with lock(root / ".latest.lock"):
                    atomic_json(
                        root / "latest.json",
                        {
                            "schema_version": 1,
                            "run_id": record["run_id"],
                            "run": f"runs/{record['run_id']}",
                            "run_sha256": sha256(self.path / "run.json"),
                            "pdf": f"runs/{record['run_id']}/rendered/homework_solutions.pdf",
                            "pdf_sha256": sha256(self.path / "rendered/homework_solutions.pdf"),
                            "validation": "engineering-only; independent style/content not implied",
                        },
                    )
            return {
                "run_id": record["run_id"],
                "run": str(self.path),
                "status": status,
                **(result or {}),
            }

    def fail(self, exc: BaseException) -> dict:
        (self.path / "logs/error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return self.finish("failed", result={"error": str(exc), "error_type": type(exc).__name__})


def verify_run(path: Path) -> dict:
    run = Run(path)
    record = run.record()
    if record["status"] == "running":
        raise ValueError("Run is incomplete")
    for directory in DIRECTORIES:
        if not inside(path, directory).is_dir():
            raise ValueError(f"Missing contract directory: {directory}")
    expected = record["artifacts"]
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()} - {"run.json"}
    if actual != set(expected):
        raise ValueError("Run artifact inventory changed")
    for name, digest in expected.items():
        if sha256(inside(path, name)) != digest:
            raise ValueError(f"Run artifact changed: {name}")
    if record.get("deliverable"):
        from hwsol.jobs.runtime import verify_job

        verify_job(path / "rendered", pixels=True)
    return {"status": "passed", "run_status": record["status"], "run_id": record["run_id"]}


def render_assignment(
    manifest: Path,
    *,
    workspace: Path | None = None,
    run_id: str | None = None,
    output: Path | None = None,
    realism: str | None = None,
    store=None,
    latest: bool = True,
) -> dict:
    from hwsol.jobs.runtime import load_job, render_job

    store = store or profile_store(workspace)
    if output and output.exists() and any(output.iterdir()):
        raise ValueError("Compatibility export must be new or empty")
    run = Run.create(
        "render",
        workspace=workspace,
        run_id=run_id,
        config={"manifest": str(manifest.resolve()), "realism_override": realism},
    )
    start = time.perf_counter()
    try:
        shutil.copyfile(manifest, run.path / "logs/request.json")
        job = load_job(manifest)
        selected = store.resolve(job.profile)
        record = run.record()
        record.update(seed=job.render.seed, profile=selected.provenance())
        record["config"].update(manifest_sha256=sha256(manifest), job=job.model_dump(mode="json"))
        atomic_json(run.path / "run.json", record)
        result = render_job(manifest, run.path / "payload", store, realism=realism)
        (run.path / "rendered/pages").rmdir()
        (run.path / "rendered").rmdir()
        (run.path / "payload").rename(run.path / "rendered")
        shutil.copytree(run.path / "rendered/solutions", run.path / "solutions", dirs_exist_ok=True)
        shutil.copyfile(
            run.path / "rendered/validation_report.json", run.path / "reports/validation.json"
        )
        result.update(
            elapsed_seconds=time.perf_counter() - start,
            pdf=str(run.path / "rendered/homework_solutions.pdf"),
        )
        if output:
            shutil.copytree(run.path / "rendered", output, dirs_exist_ok=True)
            from hwsol.jobs.runtime import verify_job

            verify_job(output, pixels=True)
            result["compatibility_export"] = str(output.resolve())
        return run.finish("passed", result=result, deliverable=latest)
    except BaseException as exc:
        if output:
            failure = run.path / "payload/validation_report.json"
            if failure.exists():
                output.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(failure, output / "validation_report.json")
            else:
                atomic_json(
                    output / "validation_report.json",
                    {
                        "status": "failed",
                        "error": str(exc),
                        "type": type(exc).__name__,
                    },
                )
        run.fail(exc)
        raise


def archive(root: Path, sources: list[Path]) -> dict:
    """Index historical data in place. Never delete, rewrite or move originals."""
    destination = workspace_root(root) / "archive" / f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
    files = {}
    for source in sources:
        source = source.resolve()
        if not source.exists():
            raise ValueError(f"Archive source missing: {source}")
        for p in sorted(source.rglob("*")) if source.is_dir() else [source]:
            if p.is_symlink():
                raise ValueError(f"Archive requires regular files, found symlink: {p}")
            if p.is_file():
                files[str(p)] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    atomic_json(destination, {"schema_version": 1, "mode": "in-place-preserved", "files": files})
    return {"index": str(destination), "files": len(files)}


def tracked_call(kind, function, *args, **kwargs):
    """Receipts for legacy Studio/lab commands; returned legacy paths remain compatible."""
    from hwsol.profiles.store import ProfileStore

    arguments = dict(inspect.signature(function).bind(*args, **kwargs).arguments)
    profile = arguments.get("profile")
    provenance = None
    if isinstance(profile, str):
        try:
            provenance = ProfileStore().resolve(profile).provenance()
        except ValueError:
            pass
    config = json.loads(json.dumps(arguments, default=str))
    checkpoints = {
        key: {"path": str(value), "sha256": sha256(Path(value))}
        for key, value in arguments.items()
        if key in {"checkpoint", "vae"} and value and Path(value).is_file()
    }
    run = Run.create(
        kind, config=config, profile=provenance, seed=arguments.get("seed"), checkpoints=checkpoints
    )
    try:
        result = function(*args, **kwargs)
        output = arguments.get("output")
        if output and Path(output).exists():
            output = Path(output)
            if output.is_dir():
                shutil.copytree(output, run.path / "comparisons/output")
            else:
                shutil.copyfile(output, run.path / "comparisons" / output.name)
        status = result.get("status", "passed") if isinstance(result, dict) else "passed"
        if status not in {"passed", "failed", "blocked", "held"}:
            status = "prepared"
        run.finish(status, result={"operation_result": result})
        return {**result, "run": str(run.path)} if isinstance(result, dict) else result
    except BaseException as exc:
        run.fail(exc)
        raise
