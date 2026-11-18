"""Immutable run receipts for corpus research; never advances production latest."""

import subprocess
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run


def create(
    kind: str,
    *,
    run_id: str | None = None,
    config: dict | None = None,
    seed: int | None = None,
    workspace: Path | None = None,
) -> Run:
    source = Path(__file__).parent
    files = {
        str(p.relative_to(source)): sha256(p)
        for p in source.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".html", ".js", ".css", ".json", ".lock"}
        and "__pycache__" not in p.parts
    }
    run = Run.create(
        kind,
        run_id=run_id,
        seed=seed,
        workspace=workspace,
        config={
            **(config or {}),
            "corpus_source_sha256": files,
            "corpus_git_status": subprocess.check_output(
                ["git", "status", "--porcelain", "--", "research/corpus"], text=True
            ),
        },
    )
    for name in files:
        dest = run.path / "logs/source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((source / name).read_bytes())
    atomic_json(
        run.path / "reports/policy.json",
        {
            "private_uploads_allowed": False,
            "automatic_verification_allowed": False,
            "production_promotion_allowed": False,
        },
    )
    return run
