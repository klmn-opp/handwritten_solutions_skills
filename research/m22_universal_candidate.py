"""Append one reviewed original universal quantifier to private CPU v2."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from hwsol.personal_ink.bundle import load_recipe
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import load_snapshot
from hwsol.workspace import Run, verify_run
from research.session_o_sources import admit, extract

R = Path("workspace/runs").resolve()
AUDIT = R / "20261120T154606311013Z-m22-math-source-gap-audit-b8673da8"
SOURCE = R / "20261120T133948143789Z-session-o-full-source-rgb-import-7e119738"
PARENT = R / "20261120T150345712055Z-portable-personal-cpu-sigma-assets-0af50b3f"


def read(p):
    return json.loads(p.read_text())


def main():
    for p in (AUDIT, SOURCE, PARENT):
        verify_run(p)
    crops = extract(SOURCE, AUDIT / "solutions/selection.json")
    with tempfile.TemporaryDirectory(prefix="hwsol-universal-approval-") as temporary:
        path = Path(temporary) / "approval.json"
        atomic_json(
            path,
            {
                "parent_run_sha256": sha256(crops / "run.json"),
                "accepted_source_ids": ["O-FULL-P10-R01-U2200-V01"],
                "review_evidence_sha256": {
                    str(AUDIT / "reports/universal-source.json"): sha256(
                        AUDIT / "reports/universal-source.json"
                    )
                },
                "independent_root_review": (
                    "Root inspected actual O forall crop/context and approved "
                    "authentic unedited ink with exact margins; instruction received "
                    "this turn. This is Agent approval, not author or production "
                    "approval."
                ),
                "historical_priority": (
                    "Eight authentic historical universal-quantifier regions inspected"
                    " first, native144DPI image. O chosen for higher native resolution"
                    " and verified300DPI PDF transform; historical candidates "
                    "retained."
                ),
            },
        )
        approved = admit(crops, path)
    run = Run.create(
        "m22-universal-private-v3",
        seed=1909,
        config={
            "parent": str(PARENT),
            "parent_sha256": sha256(PARENT / "run.json"),
            "approved_source": str(approved),
            "approved_sha256": sha256(approved / "run.json"),
            "production_activation": False,
            "neural_updates": 0,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/m22_universal_candidate.py")
        folder = run.path / "solutions/profile"
        shutil.copytree(PARENT / "solutions/profile", folder)
        manifest = read(folder / "profile.json")
        recipe = read(folder / manifest["ink_bundle"])
        bank = folder / recipe["banks"]["session_o"]
        rows = read(bank / "sources.json")
        additions = read(approved / "solutions/glyphs/sources.json")
        assert len(rows) == 36 and len(additions) == 1 and additions[0]["character"] == "∀"
        for row in additions:
            for key in ("crop", "parent_300dpi_crop"):
                target = bank / row[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists()
                shutil.copyfile(approved / "solutions/glyphs" / row[key], target)
        atomic_json(bank / "sources.json", rows + additions)
        recipe["scan_roles"] = sorted(set(recipe["scan_roles"]) | {"∀"})
        recipe["admission"]["universal_increment"] = {
            "source_run": str(approved),
            "source_sha256": sha256(approved / "run.json"),
            "prior_profile_sha256": sha256(PARENT / "solutions/profile/profile.json"),
            "author_approved": False,
            "placement_review": "pending",
        }
        atomic_json(folder / manifest["ink_bundle"], recipe)
        manifest.update(
            version="v3",
            created_at=datetime.now(UTC).isoformat(),
            parent="personal-cpu-candidate@v2",
            description=(
                "Private CPU candidate v3; approved original universal quantifier;"
                " placement gate required"
            ),
        )
        manifest["build"]["universal_increment"] = run.record()["config"]
        roles = {a["path"]: a["role"] for a in manifest["assets"]}
        manifest["assets"] = [
            {
                "path": p.relative_to(folder).as_posix(),
                "sha256": sha256(p),
                "role": roles.get(p.relative_to(folder).as_posix(), "provenance"),
            }
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p != folder / "profile.json"
        ]
        atomic_json(folder / "profile.json", manifest)
        loaded = load_snapshot(folder)
        load_recipe(loaded)
        archive = run.path / "solutions/personal-cpu-candidate-v3.zip"
        with ZipFile(archive, "x", ZIP_DEFLATED) as z:
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(folder).as_posix())
        run.finish(
            "prepared",
            result={
                "archive": str(archive),
                "archive_sha256": sha256(archive),
                "profile_reference": loaded.manifest.ref,
                "profile_sha256": loaded.manifest_sha256,
                "parent_profile": "personal-cpu-candidate@v2",
                "accepted_O_sources": 37,
                "new_sources": 1,
                "new_roles": ["∀"],
                "production_activation": False,
                "placement_validation": "pending",
            },
        )
        verify_run(run.path)
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


if __name__ == "__main__":
    main()
