"""Assemble local evidence for the reviewed v3→v4→v5 CPU increment; never publish.

The historical review anchors are deliberately fixed. A different visible result
requires a new actual review, not a command-line switch that manufactures approval.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

from hwsol.identity import digest
from hwsol.jobs.runtime import verify_job
from hwsol.personal_ink.validation import SEMANTIC_FIELDS
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.qualification import (
    AuthorReview,
    ChangeApproval,
    ExistingChange,
    GapWaiver,
    GlyphSource,
    MathReview,
    PriorReview,
    Request,
    _glyph_source,
)
from hwsol.profiles.schema import ProfileVersion
from hwsol.workspace import Run, verify_run
from research.personal_cpu_install import compare_pages

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/runs"
ANCHORS = {
    "43bb9000": "82d6a7dbfb4c957279f7451252c08d36379d661683d7f585ba7da792ae8696ac",
    "1e919407": "a1e31a0a3b7e2694168e503434484078e967429744832fdc9400a7c1122d166f",
    "ce6bbb69": "4a4c44cb93a6df81cf8351499521cb19ec7ab0a955e55b0e676fb926606b09fa",
    "8d0c2a95": "e63fa8725eb5619767044ae8800b524e19eb865da441e9e4b0a0c7ecd1504a1c",
    "67021e67": "598da60982c971cbe89ff294637f661e71ea105d126f1defc717e9b6142c0183",
    "4f709095": "7882894cf3b9ffb8f9812b6ffd1712f0e4f9e87f298584c2a54f77afd22e7e6c",
    "b95703b8": "07e6dd8c74c4d3c70354cf3ee2f2703d854c48cb13000a47df247de047e85786",
    "7be046ad": "54363b088e39c74d491ad5b11bc78036cb622f82aa964c94f83ae96b1e43e51d",
    "b43a6032": "7bb403a496a7ce834035a18cf8d78ef6f7484f340fb624639e791fd685ef0495",
}
ACCEPTED_QUOTE = "每个单词的间隔应稍微大一点点，其他整体满意"
WAIVER_QUOTE = "再加百分之8即可，后续不用验收这个了"
SCOPE = "word-gap-only-with-existing-rule-selection"


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def anchor(suffix: str) -> Path:
    matches = list(RUNS.glob(f"*-{suffix}"))
    if len(matches) != 1:
        raise ValueError(f"Historical reviewed run missing or ambiguous: {suffix}")
    verify_run(matches[0])
    if sha256(matches[0] / "run.json") != ANCHORS[suffix]:
        raise ValueError(f"Historical review anchor changed: {suffix}")
    return matches[0]


def source_signature(folder: Path) -> list:
    manifest = read(folder / "manifest.json")
    return [
        (
            problem.get("id"),
            problem["number"],
            problem.get("statement"),
            sha256(inside(folder, problem["solution"])),
            [(sha256(inside(folder, s["path"])), s.get("page")) for s in problem["sources"]],
        )
        for problem in manifest["problems"]
    ]


def exact_reviewed(expected: Path, actual: Path) -> dict:
    """Pixels, source bytes, structures and visible source choices must all match."""
    verify_job(expected, pixels=True)
    verify_job(actual, pixels=True)
    if source_signature(expected) != source_signature(actual):
        raise ValueError("Previously reviewed problem/answer bytes changed")
    rows = compare_pages(expected, actual, 4)
    before, after = (
        read(expected / "validation_report.json"),
        read(actual / "validation_report.json"),
    )
    for x, y in zip(before["pages"], after["pages"], strict=True):
        if x["matrix_groups"] != y["matrix_groups"]:
            raise ValueError("Previously reviewed matrix structure changed")
        if [_glyph_source(g) for g in x["glyph_provenance"]] != [
            _glyph_source(g) for g in y["glyph_provenance"]
        ]:
            raise ValueError("Previously reviewed original ink source changed")
    return {"pages": rows, "source_bytes_exact": True, "all_ink_sources_exact": True}


def changes(before: Path, after: Path) -> list[dict]:
    left, right = read(before / "validation_report.json"), read(after / "validation_report.json")
    if source_signature(before) != source_signature(after):
        raise ValueError("Spacing increment changed original question/answer bytes")
    found = []
    for page, (x, y) in enumerate(zip(left["pages"], right["pages"], strict=True), 1):
        if x["formula_groups"] != y["formula_groups"] or x["matrix_groups"] != y["matrix_groups"]:
            raise ValueError("Spacing increment changed formula/matrix structure")
        for item, (a, b) in enumerate(
            zip(x["glyph_provenance"], y["glyph_provenance"], strict=True)
        ):
            if [a.get(k) for k in SEMANTIC_FIELDS] != [b.get(k) for k in SEMANTIC_FIELDS]:
                raise ValueError("Spacing increment changed visible semantics")
            sa = tuple(a["bounds_px"][k + 2] - a["bounds_px"][k] for k in (0, 1))
            sb = tuple(b["bounds_px"][k + 2] - b["bounds_px"][k] for k in (0, 1))
            if (
                _glyph_source(a) == _glyph_source(b)
                and max(abs(u - v) for u, v in zip(sa, sb, strict=True)) < 0.002
            ):
                continue
            found.append(
                ExistingChange(
                    page=page,
                    item=item,
                    semantic=a["semantic"],
                    before=GlyphSource.model_validate(_glyph_source(a)),
                    after=GlyphSource.model_validate(_glyph_source(b)),
                    before_size=sa,
                    after_size=sb,
                ).model_dump(mode="json")
            )
    return found


def assemble(args) -> Path:
    for path in (args.linux, args.mac):
        verify_run(path)
    historical = {suffix: anchor(suffix) for suffix in ANCHORS}
    # These checks bind the reusable RC1 outputs to actual independently reviewed
    # pages, including the failed-first-comparison history rather than relabelling it.
    exact_reviewed(
        historical["4f709095"] / "rendered/v5",
        historical["8d0c2a95"] / "rendered/python-3.13/candidate",
    )
    native_review = read(historical["b95703b8"] / "run.json")["result"]
    if not (
        native_review.get("native_v5_functional") == "passed"
        and native_review.get("native_v4_to_v5_f_source_changes") == 2
        and native_review.get("visual_review") == "no newly confirmed content/math/boundary defect"
    ):
        raise ValueError("The actual native independent review is absent or changed")
    author_text = args.author_record.read_text()
    if ACCEPTED_QUOTE not in author_text or WAIVER_QUOTE not in author_text:
        raise ValueError("Actual accepted-base/spacing author statements are missing")
    facts = read(historical["7be046ad"] / "reports/independent-checks.json")["facts"]
    if len(facts) != 16 or not all(value is True for value in facts.values()):
        raise ValueError("The actual independent mathematical checks are not all passed")
    exact_reviewed(historical["43bb9000"] / "rendered/python-3.12/candidate", args.base_v3_job)
    run = Run.create(
        "m23-local-qualification-bundle",
        seed=1909,
        config={
            "purpose": "assemble actual reviewed evidence; no qualify, publish or activation",
            "linux": str(args.linux),
            "mac": str(args.mac),
            "historical_review_run_shas": ANCHORS,
            "author_record_sha256": sha256(args.author_record),
        },
    )
    print(run.path, flush=True)
    root = run.path / "solutions/qualification"
    root.mkdir()
    shutil.copyfile(__file__, run.path / "logs/m23_qualification_bundle.py")
    try:

        def copied(source: Path, name: str) -> dict:
            target = inside(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256(target) != sha256(source):
                raise ValueError("Evidence copy changed")
            return {"path": name, "sha256": sha256(target)}

        def written(name: str, value: dict, model=None) -> dict:
            if model:
                value = model.model_validate(value).model_dump(mode="json", by_alias=True)
            atomic_json(inside(root, name), value)
            return {"path": name, "sha256": sha256(inside(root, name))}

        def job(source: Path, name: str) -> dict:
            verify_job(source, pixels=True)
            target = inside(root, name)
            shutil.copytree(source, target)
            verify_job(target, pixels=True)
            return {
                "directory": name,
                "report": {
                    "path": f"{name}/validation_report.json",
                    "sha256": sha256(target / "validation_report.json"),
                },
            }

        def profile(archive: Path, label: str) -> tuple[dict, ProfileVersion]:
            copied(archive, f"profiles/{label}.zip")
            with ZipFile(archive) as z:
                raw = z.read("profile.json")
            target = root / f"profiles/{label}.json"
            target.write_bytes(raw)
            return {
                "path": target.relative_to(root).as_posix(),
                "sha256": sha256(target),
            }, ProfileVersion.model_validate_json(raw)

        base_file, base = profile(args.base_v3_zip, "accepted-v3")
        prior_file, prior = profile(args.prior_v4_zip, "prescribed-v4")
        candidate_file, candidate = profile(args.candidate_zip, "final-v5")
        candidate_sha = candidate_file["sha256"]
        if (base.version, prior.version, candidate.version) != ("v3", "v4", "v5"):
            raise ValueError("This assembler only covers the actually reviewed v3/v4/v5 lineage")
        old_render = prior.render.model_dump()
        new_render = candidate.render.model_dump()
        base_render = base.render.model_dump()
        if not (
            old_render.pop("word_gap_scale") == 1.06
            and new_render.pop("word_gap_scale") == 1.1448
            and base_render.pop("word_gap_scale") == 1.0
            and old_render == new_render == base_render
            and prior.parent == base.ref
            and candidate.parent == prior.ref
            and prior.assets == candidate.assets == base.assets
            and prior.ink_bundle == candidate.ink_bundle == base.ink_bundle
        ):
            raise ValueError("Profile assets/lineage differ beyond the prescribed spacing")
        base_job = job(args.base_v3_job, "accepted-base/job")
        if (
            read(args.base_v3_job / "validation_report.json")["profile"]["manifest_sha256"]
            != base_file["sha256"]
        ):
            raise ValueError("Author-approved base Job and Profile differ")
        copied(args.author_record, "history/author-record.md")
        for suffix, source in historical.items():
            copied(source / "run.json", f"history/{suffix}/run-receipt.json")
        for suffix in ("7be046ad", "b43a6032", "4f709095", "b95703b8"):
            for source in sorted((historical[suffix] / "reports").rglob("*")):
                if source.is_file():
                    copied(
                        source,
                        "history/"
                        + suffix
                        + "/reports/"
                        + source.relative_to(historical[suffix] / "reports").as_posix(),
                    )
        wheel = next((args.linux / "reports/dist").glob("*.whl"))
        wheel_file = copied(wheel, "build/" + wheel.name)
        installation = copied(args.linux / "reports/installations.json", "installation/linux.json")
        platforms, deltas, comparisons, reports = [], [], [], []
        build = None
        for system, current, old_final, old_prior in (
            ("Linux", args.linux, historical["8d0c2a95"], historical["1e919407"]),
            ("Darwin", args.mac, historical["67021e67"], historical["ce6bbb69"]),
        ):
            relative = "rendered" if system == "Linux" else "rendered/native"
            for version in ("3.12", "3.13"):
                label = f"{system.lower()}-{version}"
                folder = current / relative / f"python-{version}"
                reviewed = old_final / relative / f"python-{version}/candidate"
                prior_reviewed = old_prior / relative / f"python-{version}/candidate"
                receipt = read(folder / "receipt.json")
                environment = receipt["qualification_environment"]
                if (
                    receipt["status"] != "passed"
                    or environment["system"] != system
                    or environment["python_minor"] != version
                ):
                    raise ValueError(
                        "Native qualification environment does not match actual receipt"
                    )
                if receipt["inputs"]["wheel"]["sha256"] != wheel_file["sha256"]:
                    raise ValueError("Native wheel differs from exact Linux wheel")
                evidence_job, prior_job = folder / "candidate", folder / "rollback"
                comparisons.append(
                    {
                        "environment": environment,
                        "final": exact_reviewed(reviewed, evidence_job),
                        "prior": exact_reviewed(prior_reviewed, prior_job),
                    }
                )
                expected_changes = changes(prior_reviewed, reviewed)
                actual_changes = changes(prior_job, evidence_job)
                if actual_changes != expected_changes:
                    raise ValueError("Unexpected delta; a new independent review is required")
                job_ref, prior_ref = (
                    job(evidence_job, f"platforms/{label}/job"),
                    job(prior_job, f"platforms/{label}/prior-job"),
                )
                job(reviewed, f"reviewed/{label}/job")
                job(prior_reviewed, f"reviewed/{label}/prior-job")
                report = read(evidence_job / "validation_report.json")
                if report["profile"]["manifest_sha256"] != candidate_sha:
                    raise ValueError("Complete Job does not use selected v5 Profile")
                identity = report["build"]
                this_build = {
                    "commit": identity["commit"],
                    "version": identity["packages"]["handwritten-solutions"],
                    "lock_sha256": identity["lock_sha256"],
                    "source_sha256": digest(identity["source"]),
                    "handnote_source_sha256": digest(identity["handnote_source"]),
                    "wheel": wheel_file,
                }
                if build is not None and build != this_build:
                    raise ValueError("Platform code, wheel or pinned Handnote differs")
                build = this_build
                row = {
                    "environment": environment,
                    "renderer_fingerprint": identity["renderer_fingerprint"],
                    "receipt": copied(folder / "receipt.json", f"platforms/{label}/portable.json"),
                    "job": job_ref,
                }
                if system == "Linux":
                    row.update(installation=installation, installation_portable=row["receipt"])
                elif receipt.get("clean_install_completed") is not True:
                    raise ValueError(
                        "This assembly requires a genuinely clean native Mac candidate run"
                    )
                platforms.append(row)
                reports.append(job_ref["report"]["sha256"])
                review = written(
                    f"reviews/{label}-change.json",
                    {
                        "schema": "personal-cpu-layout-change-review/1",
                        "environment": environment,
                        "profile_sha256": candidate_sha,
                        "prior_profile_sha256": prior_file["sha256"],
                        "before_report_sha256": prior_ref["report"]["sha256"],
                        "after_report_sha256": job_ref["report"]["sha256"],
                        "independent_review_passed": True,
                        "statement": (
                            "Exact output/source/structure matches the actual previously "
                            "reviewed per-platform v4/v5 pages; all changes equal that review. "
                            "Historical review records and complete comparison pages are "
                            "retained. No new visual approval is inferred from training or hash "
                            "equality alone."
                        ),
                        "accepted_changes": expected_changes,
                    },
                    ChangeApproval,
                )
                deltas.append(
                    {"environment": environment, "prior_job": prior_ref, "review": review}
                )
        prior_review = written(
            "reviews/prior-author.json",
            {
                "schema": "personal-cpu-prior-review/1",
                "basis": "accepted-base-plus-prescribed-spacing",
                "author_accepted_base": True,
                "base_profile": base_file,
                "base_job": base_job,
                "prescribed_parent_profile_sha256": prior_file["sha256"],
                "statement": (
                    f"Actual author statement on accepted v3 pages: {ACCEPTED_QUOTE}. "
                    f"The subsequent v4→v5 spacing instruction is {WAIVER_QUOTE}. "
                    "Full source record is history/author-record.md; this does not claim "
                    "the author directly viewed new RC2 pages."
                ),
            },
            PriorReview,
        )
        gap = written(
            "reviews/spacing-waiver.json",
            {
                "schema": "personal-cpu-gap-waiver/1",
                "scope": SCOPE,
                "profile_sha256": candidate_sha,
                "prior_profile": prior_file,
                "before_scale": 1.06,
                "after_scale": 1.1448,
                "increment": 1.08,
                "changes": deltas,
                "job_report_sha256": reports,
            },
            GapWaiver,
        )
        author = written(
            "reviews/author.json",
            {
                "schema": "personal-cpu-author-review/1",
                "profile_sha256": candidate_sha,
                "kind": "scoped-change-waiver",
                "approved": True,
                "job_report_sha256": reports,
                "prior_review": prior_review,
                "change_report": gap,
                "scope": SCOPE,
                "statement": (
                    "Scope limited to actual accepted v3 style plus prescribed spacing: "
                    f"{ACCEPTED_QUOTE}; {WAIVER_QUOTE}. "
                    "New package outputs must exactly match those independently reviewed "
                    "pages; no production activation/publication is authorized by this receipt."
                ),
            },
            AuthorReview,
        )
        math = written(
            "reviews/mathematics.json",
            {
                "schema": "personal-cpu-math-review/1",
                "independent_math_verified": True,
                "job_report_sha256": reports,
                "statement": (
                    "Actual independent Agent derivations and16 symbolic checks are "
                    "retained from7be046ad and auditb43a6032. Original question/answer "
                    "bytes and mathematical structures match the previously verified "
                    "complete Jobs on every platform. Rendering is not used as a "
                    "substitute mathematical proof."
                ),
            },
            MathReview,
        )
        written("reviews/exact-reviewed-comparisons.json", {"comparisons": comparisons})
        files = {
            p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
        request = Request.model_validate(
            {
                "schema": "personal-cpu-qualification-request/1",
                "profile": candidate.ref,
                "profile_sha256": candidate_sha,
                "recipe_sha256": next(
                    a.sha256 for a in candidate.assets if a.path == candidate.ink_bundle
                ),
                "build": build,
                "platforms": platforms,
                "author": author,
                "math_review": math,
                "files": files,
            }
        )
        if sum(inside(root, p).stat().st_size for p in files) > 512 * 1024**2:
            raise ValueError("Evidence exceeds the runtime qualification limit")
        atomic_json(root / "request.json", request.model_dump(mode="json", by_alias=True))
        run.finish(
            "held",
            result={
                "request": str(root / "request.json"),
                "request_sha256": sha256(root / "request.json"),
                "platform_environments": len(platforms),
                "reviewed_output_identity": "exact",
                "qualification_executed": False,
                "published": False,
                "activated": False,
            },
        )
        verify_run(run.path)
        return run.path
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("linux", "mac", "base-v3-job", "base-v3-zip", "prior-v4-zip", "candidate-zip"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--author-record", type=Path, default=ROOT / "docs/M22_WORD_GAP_REVISION.md"
    )
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    print(assemble(args))


if __name__ == "__main__":
    main()
