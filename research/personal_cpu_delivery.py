"""Seal reviewed CPU integration evidence without regenerating accepted ink."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from hwsol.jobs.runtime import verify_job
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.workspace import Run, verify_run


def read(path: Path):
    return json.loads(path.read_text("utf-8"))


def package(specification: Path) -> Path:
    spec = read(specification)
    parents = {}
    for role, entry in spec["parents"].items():
        path = Path(entry["path"])
        verify_run(path)
        if sha256(path / "run.json") != entry["sha256"]:
            raise ValueError(f"Reviewed evidence changed: {role}")
        parents[role] = path
    new = read(parents["new"] / "reports/summary.json")
    regression = read(parents["regression"] / "reports/summary.json")
    stress = read(parents["stress"] / "reports/summary.json")
    installation = read(parents["install"] / "reports/result.json")
    rollback = read(parents["rollback"] / "reports/result.json")
    if not (
        new["rendered"] == new["expectations_met"] == 6
        and not new["font_fallbacks"]
        and regression["rendered"] == regression["regression_pixel_equal"] == 24
        and regression["regression_identity_equal"] == 24
        and not regression["font_fallbacks"]
        and stress["expectations_met"] == 10
        and stress["rendered"] == stress["rejected"] == 5
        and installation["linux_python_3_12_3_13"] == "passed"
        and installation["all_reference_and_cross_python_pixels_exact"]
        and installation["source_commit"] == spec["runtime_source_commit"]
        and rollback["rollback_v1_pixels_exact"]
        and rollback["production_defaults_settings_latest_unchanged"]
    ):
        raise ValueError("Integration evidence does not meet the frozen bounded stage gates")
    for role, review in spec["reviews"].items():
        if sha256(Path(review["path"])) != review["sha256"]:
            raise ValueError(f"Review changed: {role}")
    for path, digest in spec["protected"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Protected asset changed: {path}")
    accepted = parents["combined"] / "rendered"
    verify_job(accepted, pixels=True)
    render_commit = read(accepted / "validation_report.json")["build"]["commit"]
    run = Run.create(
        "personal-cpu-integration-reviewed-delivery",
        seed=1909,
        config={
            "purpose": "package six accepted pages and integration evidence; no new generation",
            "specification_sha256": sha256(specification),
            "runtime_source_commit": spec["runtime_source_commit"],
            "copied_render_source_commit": render_commit,
            "parents": spec["parents"],
            "production_activation": False,
            "neural_parameter_updates": 0,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/personal_cpu_delivery.py")
        shutil.copyfile(specification, run.path / "reports/reviewed-inputs.json")
        shutil.copytree(accepted, run.path / "rendered", dirs_exist_ok=True)
        output_files = []
        for source in sorted(p for p in accepted.rglob("*") if p.is_file()):
            name = source.relative_to(accepted).as_posix()
            if sha256(source) != sha256(inside(run.path / "rendered", name)):
                raise ValueError(f"Copied accepted output changed: {name}")
            output_files.append({"path": name, "sha256": sha256(source)})
        verify_job(run.path / "rendered", pixels=True)
        with ZipFile(run.path / "comparisons/pages-300dpi.zip", "x", ZIP_DEFLATED) as archive:
            for page in sorted((run.path / "rendered/pages").glob("*.png")):
                archive.write(page, page.name)
        assets = read(parents["assets"] / "reports/result.json")
        private_archive = Path(assets["archive"])
        if sha256(private_archive) != assets["archive_sha256"]:
            raise ValueError("Private candidate archive changed")
        shutil.copyfile(private_archive, run.path / "solutions" / private_archive.name)
        for role, review in spec["reviews"].items():
            source = Path(review["path"])
            shutil.copyfile(source, run.path / "reports" / f"{role}{source.suffix}")
        for role, parent in parents.items():
            destination = run.path / "reports/parents" / role
            destination.mkdir(parents=True)
            shutil.copyfile(parent / "run.json", destination / "receipt.json")
            for name in ("result.json", "summary.json"):
                if (parent / "reports" / name).is_file():
                    shutil.copyfile(parent / "reports" / name, destination / name)
        atomic_json(run.path / "reports/copied-output-sha256.json", output_files)
        atomic_json(run.path / "reports/protected-assets.json", spec["protected"])
        summary = {
            "stage": "portable reviewed personal CPU integration",
            "runtime_source_commit": spec["runtime_source_commit"],
            "copied_render_source_commit": render_commit,
            "profile": assets["profile_reference"],
            "profile_sha256": assets["profile_sha256"],
            "private_archive_sha256": assets["archive_sha256"],
            "new_complete_pages": 6,
            "new_visible_math_items": new["math_visible_items"],
            "new_page_font_fallbacks": 0,
            "accepted_regression_pages_pixel_exact": 24,
            "regression_visible_math_items": regression["math_visible_items"],
            "stress_font_fallbacks": stress["font_fallbacks"],
            "explicit_rejected_inputs": stress["rejected"],
            "linux_installed_python_3_12_3_13": "passed",
            "macos": "not executed on this Linux host",
            "candidate_stage_go": True,
            "production_release_go": False,
            "author_feedback": "broad satisfaction with preceding24; no formal blind claim",
            "new_neural_parameter_updates": 0,
            "unchanged_accepted_job_bytes": len(output_files),
            "limits": spec["remaining_limits"],
        }
        for path, digest in spec["protected"].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Protected asset changed during packaging: {path}")
        verify_job(run.path / "rendered", pixels=True)
        run.finish("held", result=summary)
        verify_run(run.path)
        return run.path
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    args = parser.parse_args()
    print(package(args.specification))


if __name__ == "__main__":
    main()
