"""Local, build-bound personal CPU qualification; never trust imported recipe claims.

These receipts prove local evidence consistency, not the truth of an owner's
attestation against an adversary who can rewrite the entire local store. They are
not portable publication authority and do not activate a Profile.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

from pydantic import Field, StrictBool, field_validator, model_validator

from hwsol.identity import build_identity, digest, runtime_text_capabilities
from hwsol.profiles.io import atomic_json, inside, lock, sha256
from hwsol.profiles.schema import (
    Digest,
    FrozenModel,
    ProfileError,
    ProfileVersion,
    reference,
    relative,
)
from hwsol.profiles.store import copy_snapshot


class File(FrozenModel):
    path: str
    sha256: Digest
    _path = field_validator("path")(relative)


class Library(FrozenModel):
    enabled: StrictBool
    version: str | None


class Libraries(FrozenModel):
    raqm: Library
    harfbuzz: Library
    fribidi: Library
    freetype2: Library


class Environment(FrozenModel):
    system: Literal["Linux", "Darwin"]
    machine: str = Field(min_length=1, max_length=64)
    python_minor: str = Field(pattern=r"^3\.\d+$")
    pillow_version: str = Field(min_length=1)
    layout_engine: Literal["BASIC", "RAQM"]
    libraries: Libraries


class Build(FrozenModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    version: str = Field(min_length=1)
    lock_sha256: Digest
    source_sha256: Digest
    handnote_source_sha256: Digest
    wheel: File


class JobEvidence(FrozenModel):
    directory: str
    report: File
    _directory = field_validator("directory")(relative)

    @model_validator(mode="after")
    def report_path(self):
        if self.report.path != f"{self.directory}/validation_report.json":
            raise ValueError("Job report must be inside its declared Job")
        return self


class PlatformEvidence(FrozenModel):
    environment: Environment
    renderer_fingerprint: Digest
    receipt: File
    job: JobEvidence
    installation: File | None = None
    installation_portable: File | None = None


class AuthorReview(FrozenModel):
    schema_: Literal["personal-cpu-author-review/1"] = Field(alias="schema")
    profile_sha256: Digest
    kind: Literal["direct-pages", "scoped-change-waiver"]
    approved: StrictBool
    statement: str = Field(min_length=1, max_length=4000)
    job_report_sha256: list[Digest] = Field(min_length=1, max_length=16)
    prior_review: File | None = None
    change_report: File | None = None
    scope: Literal["word-gap-only-with-existing-rule-selection"] | None = None

    @model_validator(mode="after")
    def waiver(self):
        if self.kind == "scoped-change-waiver" and not (
            self.prior_review and self.change_report and self.scope
        ):
            raise ValueError("Scoped waiver requires prior review, exact change report and scope")
        if self.kind == "direct-pages" and any((self.prior_review, self.change_report, self.scope)):
            raise ValueError("Direct review cannot disguise a scoped waiver")
        return self


class PriorReview(FrozenModel):
    """Author accepted the base style, then prescribed revisions from its descendant."""

    schema_: Literal["personal-cpu-prior-review/1"] = Field(alias="schema")
    basis: Literal["accepted-base-plus-prescribed-spacing"]
    author_accepted_base: StrictBool
    base_profile: File
    base_job: JobEvidence
    prescribed_parent_profile_sha256: Digest
    statement: str = Field(min_length=1)


class GlyphSource(FrozenModel):
    source_type: Literal["personal-outline", "personal-outline-geometric-calibration"]
    actual_glyph_id: int | None
    glyph_name: str | None
    variant: int | None
    font_sha256: Digest


class ExistingChange(FrozenModel):
    page: int = Field(ge=1)
    item: int = Field(ge=0)
    semantic: str = Field(min_length=1, max_length=1)
    before: GlyphSource
    after: GlyphSource
    before_size: tuple[float, float]
    after_size: tuple[float, float]


class ChangeApproval(FrozenModel):
    schema_: Literal["personal-cpu-layout-change-review/1"] = Field(alias="schema")
    environment: Environment
    profile_sha256: Digest
    prior_profile_sha256: Digest
    before_report_sha256: Digest
    after_report_sha256: Digest
    independent_review_passed: StrictBool
    statement: str = Field(min_length=1)
    accepted_changes: list[ExistingChange] = Field(max_length=32)


class PlatformChange(FrozenModel):
    environment: Environment
    prior_job: JobEvidence
    review: File


class GapWaiver(FrozenModel):
    """Specific author increment; each native environment has its own reviewed baseline."""

    schema_: Literal["personal-cpu-gap-waiver/1"] = Field(alias="schema")
    scope: Literal["word-gap-only-with-existing-rule-selection"]
    profile_sha256: Digest
    prior_profile: File
    before_scale: Literal[1.06]
    after_scale: Literal[1.1448]
    increment: Literal[1.08]
    changes: list[PlatformChange] = Field(min_length=2, max_length=16)
    job_report_sha256: list[Digest] = Field(min_length=1, max_length=16)


class MathReview(FrozenModel):
    schema_: Literal["personal-cpu-math-review/1"] = Field(alias="schema")
    independent_math_verified: StrictBool
    job_report_sha256: list[Digest] = Field(min_length=1, max_length=16)
    statement: str = Field(min_length=1)


class Request(FrozenModel):
    schema_: Literal["personal-cpu-qualification-request/1"] = Field(alias="schema")
    profile: str
    profile_sha256: Digest
    recipe_sha256: Digest
    build: Build
    platforms: list[PlatformEvidence] = Field(min_length=2, max_length=16)
    author: File
    math_review: File
    files: dict[str, Digest] = Field(min_length=1, max_length=2048)
    _profile = field_validator("profile")(reference)

    @field_validator("files")
    @classmethod
    def file_paths(cls, values):
        if len({p.casefold() for p in values}) != len(values):
            raise ValueError("Duplicate evidence paths")
        for path in values:
            relative(path)
        return values


class Receipt(FrozenModel):
    schema_: Literal["personal-cpu-qualification/1"] = Field(alias="schema")
    qualification_id: Digest
    profile: str
    profile_sha256: Digest
    request_sha256: Digest
    created_at: str
    status: Literal["locally-qualified"]
    activation: Literal[False] = False
    _profile = field_validator("profile")(reference)


class Publication(FrozenModel):
    schema_: Literal["personal-cpu-publication/1"] = Field(alias="schema")
    profile: str
    profile_sha256: Digest
    qualification_id: Digest
    published_at: str
    activated: Literal[False] = False
    _profile = field_validator("profile")(reference)


def _json(path):
    return json.loads(path.read_text("utf-8"))


def _file(root, request, evidence):
    if request.files.get(evidence.path) != evidence.sha256:
        raise ProfileError("Evidence reference is missing from its hashed inventory")
    target = inside(root, evidence.path)
    if not target.is_file() or sha256(target) != evidence.sha256:
        raise ProfileError(f"Evidence absent or changed: {evidence.path}")
    return target


def _record(root, request, evidence):
    return _json(_file(root, request, evidence))


def _build_matches(value, expected):
    return (
        value.get("working_tree_dirty") is False
        and value.get("commit") == expected.commit
        and value.get("packages", {}).get("handwritten-solutions") == expected.version
        and value.get("lock_sha256") == expected.lock_sha256
        and digest(value.get("source")) == expected.source_sha256
        and digest(value.get("handnote_source")) == expected.handnote_source_sha256
    )


def _native_core(row, spec, evidence):
    return (
        row.get("schema") == "cross-platform-candidate-check/1"
        and row.get("status") == "passed"
        and row.get("actual_system") == evidence.environment.system
        and row.get("source_commit") == spec.commit
        and row.get("package_version") == spec.version
        and row.get("inputs", {}).get("wheel", {}).get("sha256") == spec.wheel.sha256
        and type(row.get("installed_wheel_members_exact")) is int
        and row["installed_wheel_members_exact"] > 0
    )


def _clean_install(root, request, evidence, row):
    if row.get("clean_install_completed") is True:
        return True
    if not evidence.installation:
        return False
    parent = _record(root, request, evidence.installation)
    external = None
    if isinstance(parent, list):
        matches = [p for p in parent if p.get("python") == evidence.environment.python_minor]
        if len(matches) != 1 or not evidence.installation_portable:
            return False
        external = matches[0]
        if not (
            external.get("status") == "passed"
            and external.get("clean_install_completed") is True
            and external.get("source_commit") == request.build.commit
            and external.get("package_version") == request.build.version
            and external.get("wheel_sha256") == request.build.wheel.sha256
            and external.get("portable_receipt_sha256") == evidence.installation_portable.sha256
        ):
            return False
        parent = _record(root, request, evidence.installation_portable)
        location, fresh = parent.get("runtime", {}).get("hwsol"), external.get("fresh_environment")
        if not location or not fresh or not Path(location).is_relative_to(Path(fresh)):
            return False
    return (
        _native_core(parent, request.build, evidence)
        and (external is not None or parent.get("clean_install_completed") is True)
        and parent.get("runtime") == row.get("runtime")
        and parent.get("qualification_environment") == row.get("qualification_environment")
    )


def _wheel(root, request):
    path = _file(root, request, request.build.wheel)
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ProfileError("Duplicate wheel member")
        meta = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(meta) != 1:
            raise ProfileError("Ambiguous wheel metadata")
        metadata = archive.read(meta[0]).decode().splitlines()
        version = next(
            (v[9:] for v in metadata if v.startswith("Version: ")),
            None,
        )
        build = json.loads(archive.read("hwsol/_build.json"))
        source = {
            n[6:]: hashlib.sha256(archive.read(n)).hexdigest()
            for n in names
            if n.startswith("hwsol/") and n.endswith(".py")
        }
        if not (
            "Name: handwritten-solutions" in metadata
            and version == request.build.version
            and build.get("commit") == request.build.commit
            and build.get("lock_sha256") == request.build.lock_sha256
            and digest(source) == request.build.source_sha256
        ):
            raise ProfileError("Actual wheel code/metadata differs from the qualified build")


def _glyph_source(row):
    value = {
        k: row.get(k)
        for k in ("source_type", "actual_glyph_id", "glyph_name", "variant", "font_sha256")
    }
    for name in ("scan_source", "session_source", "session_o_source"):
        if isinstance(row.get(name), dict):
            value[name] = {
                k: row[name].get(k) for k in ("id", "source_id", "crop_sha256", "character")
            }
    return value


def _validate_waiver(loaded, request, root, author, reports, *, pixels):
    from hwsol.jobs.runtime import verify_job
    from hwsol.personal_ink.validation import SEMANTIC_FIELDS

    prior = PriorReview.model_validate(_record(root, request, author.prior_review))
    change = GapWaiver.model_validate(_record(root, request, author.change_report))
    old = ProfileVersion.model_validate_json(
        _file(root, request, change.prior_profile).read_bytes()
    )
    base = ProfileVersion.model_validate_json(_file(root, request, prior.base_profile).read_bytes())
    basis_report = _record(root, request, prior.base_job.report)
    verify_job(inside(root, prior.base_job.directory), pixels=pixels)
    base_render, old_render = base.render.model_dump(), old.render.model_dump()
    base_render.pop("word_gap_scale")
    old_render.pop("word_gap_scale")
    if not (
        prior.author_accepted_base
        and prior.prescribed_parent_profile_sha256 == change.prior_profile.sha256
        and basis_report.get("profile", {}).get("manifest_sha256") == prior.base_profile.sha256
        and old.parent == base.ref
        and old.assets == base.assets
        and old.font == base.font
        and old.variants == base.variants
        and old.coverage == base.coverage
        and old.ink_bundle == base.ink_bundle
        and base_render == old_render
        and old.ref == loaded.manifest.parent
        and change.profile_sha256 == loaded.manifest_sha256
        and set(change.job_report_sha256) == set(reports)
    ):
        raise ProfileError(
            "Scoped waiver accepted base/prescribed parent or exact evidence differs"
        )
    a, b = old.render.model_dump(), loaded.manifest.render.model_dump()
    if not (
        a.pop("word_gap_scale") == change.before_scale
        and b.pop("word_gap_scale") == change.after_scale
        and a == b
        and old.assets == loaded.manifest.assets
        and old.font == loaded.manifest.font
        and old.variants == loaded.manifest.variants
        and old.coverage == loaded.manifest.coverage
        and old.ink_bundle == loaded.manifest.ink_bundle
    ):
        raise ProfileError("Scoped waiver permits only the prescribed word-gap change")
    change_map = {digest(c.environment.model_dump()): c for c in change.changes}
    expected = {digest(p.environment.model_dump()) for p in request.platforms}
    if set(change_map) != expected or len(change_map) != len(change.changes):
        raise ProfileError("Scoped waiver must cover each exact native environment once")
    asset_hashes = {a.sha256 for a in loaded.manifest.assets}
    for evidence in request.platforms:
        delta = change_map[digest(evidence.environment.model_dump())]
        approval = ChangeApproval.model_validate(_record(root, request, delta.review))
        if not (
            approval.environment == evidence.environment
            and approval.profile_sha256 == loaded.manifest_sha256
            and approval.prior_profile_sha256 == change.prior_profile.sha256
            and approval.before_report_sha256 == delta.prior_job.report.sha256
            and approval.after_report_sha256 == evidence.job.report.sha256
            and approval.independent_review_passed
        ):
            raise ProfileError("Native baseline/change review is missing or stale")
        before = _record(root, request, delta.prior_job.report)
        verify_job(inside(root, delta.prior_job.directory), pixels=pixels)
        if before.get("profile", {}).get("manifest_sha256") != change.prior_profile.sha256:
            raise ProfileError("Prior native Job belongs to a different Profile")
        after = _record(root, request, evidence.job.report)
        if len(before["pages"]) != len(after["pages"]):
            raise ProfileError("Scoped waiver changed the reviewed page inventory")

        # Match actual source bytes as well as visible output. Native platform
        # layouts may differ; that does not permit changing the reviewed answers.
        def source_signature(directory):
            job = _json(inside(root, directory) / "manifest.json")
            return [
                (
                    p.get("id"),
                    p["number"],
                    p.get("statement"),
                    sha256(inside(inside(root, directory), p["solution"])),
                    [(src.get("sha256"), src.get("page")) for src in p["sources"]],
                )
                for p in job["problems"]
            ]

        if not (
            source_signature(prior.base_job.directory)
            == source_signature(delta.prior_job.directory)
            == source_signature(evidence.job.directory)
        ):
            raise ProfileError("Scoped waiver changed the reviewed source question or answer bytes")
        observed = []
        for number, (left, right) in enumerate(
            zip(before["pages"], after["pages"], strict=True), 1
        ):
            x, y = left["glyph_provenance"], right["glyph_provenance"]
            if [[r.get(k) for k in SEMANTIC_FIELDS] for r in x] != [
                [r.get(k) for k in SEMANTIC_FIELDS] for r in y
            ]:
                raise ProfileError("Scoped waiver changed visible semantics")
            for field, keys in (
                ("formula_groups", ("latex", "geometry_sha256", "items")),
                (
                    "matrix_groups",
                    ("environment", "prefix", "rows", "items", "cell_and_matrix_size_px"),
                ),
            ):
                if [[g[k] for k in keys] for g in left[field]] != [
                    [g[k] for k in keys] for g in right[field]
                ]:
                    raise ProfileError("Scoped waiver changed native mathematical structure")
            for i, (u, v) in enumerate(zip(x, y, strict=True)):
                u_size = tuple(u["bounds_px"][k + 2] - u["bounds_px"][k] for k in (0, 1))
                v_size = tuple(v["bounds_px"][k + 2] - v["bounds_px"][k] for k in (0, 1))
                if (
                    _glyph_source(u) == _glyph_source(v)
                    and max(abs(a - b) for a, b in zip(u_size, v_size, strict=True)) < 0.002
                ):
                    continue
                if not (
                    u.get("context") == v.get("context") == "text"
                    and u.get("font_sha256") in asset_hashes
                    and v.get("font_sha256") in asset_hashes
                ):
                    raise ProfileError(
                        "Scoped waiver changed math ink or introduced an unregistered source"
                    )
                observed.append(
                    ExistingChange(
                        page=number,
                        item=i,
                        semantic=u["semantic"],
                        before=GlyphSource.model_validate(_glyph_source(u)),
                        after=GlyphSource.model_validate(_glyph_source(v)),
                        before_size=u_size,
                        after_size=v_size,
                    )
                )
        if len(observed) != len(approval.accepted_changes):
            raise ProfileError("Scoped waiver has unreviewed existing-rule changes")
        for actual, accepted in zip(observed, approval.accepted_changes, strict=True):
            if actual.model_dump(exclude={"before_size", "after_size"}) != accepted.model_dump(
                exclude={"before_size", "after_size"}
            ) or any(
                abs(a - b) > 0.002
                for key in ("before_size", "after_size")
                for a, b in zip(getattr(actual, key), getattr(accepted, key), strict=True)
            ):
                raise ProfileError("Scoped waiver existing-rule change differs from review")


def _validate(loaded, request, evidence_root, *, pixels):
    from hwsol.jobs.runtime import verify_job
    from hwsol.personal_ink.bundle import load_recipe

    recipe = load_recipe(loaded)
    if (
        request.profile != loaded.manifest.ref
        or request.profile_sha256 != loaded.manifest_sha256
        or request.recipe_sha256 != recipe.sha256
    ):
        raise ProfileError("Qualification Profile or recipe identity is stale")
    total = 0
    for name, expected in request.files.items():
        path = _file(evidence_root, request, File(path=name, sha256=expected))
        total += path.stat().st_size
        if total > 512 * 1024**2:
            raise ProfileError("Qualification evidence exceeds 512 MiB")
    current = build_identity()
    if not _build_matches(current, request.build):
        raise ProfileError("Qualification build is stale or working source is dirty")
    environment = Environment.model_validate(runtime_text_capabilities(loaded.font))
    targets = [p.environment.model_dump() for p in request.platforms]
    if len({digest(t) for t in targets}) != len(targets):
        raise ProfileError("Duplicate qualified environment")
    if {p.environment.system for p in request.platforms} != {"Linux", "Darwin"}:
        raise ProfileError("Both actual Linux and Darwin evidence are required")
    matching = [p for p in request.platforms if p.environment == environment]
    if len(matching) != 1 or matching[0].renderer_fingerprint != current["renderer_fingerprint"]:
        raise ProfileError("Current text capabilities or renderer are not qualified")
    _wheel(evidence_root, request)
    reports = []
    for platform in request.platforms:
        row = _record(evidence_root, request, platform.receipt)
        if not (
            _native_core(row, request.build, platform)
            and row.get("qualification_environment") == platform.environment.model_dump()
            and _clean_install(evidence_root, request, platform, row)
        ):
            raise ProfileError("Native platform or clean installation proof is invalid")
        job_root = inside(evidence_root, platform.job.directory)
        for path in job_root.rglob("*"):
            if path.is_file():
                name = path.relative_to(evidence_root).as_posix()
                if name not in request.files:
                    raise ProfileError("Job has evidence outside the qualification inventory")
        report = _record(evidence_root, request, platform.job.report)
        verify_job(job_root, pixels=pixels)
        manifest = _json(job_root / "manifest.json")
        if not (
            manifest.get("profile") == loaded.manifest.ref
            and manifest.get("profile_sha256") == loaded.manifest_sha256
            and report.get("profile", {}).get("manifest_sha256") == loaded.manifest_sha256
            and _build_matches(report.get("build", {}), request.build)
            and report["build"].get("renderer_fingerprint") == platform.renderer_fingerprint
            and any(
                o.get("report_sha256") == platform.job.report.sha256
                and o.get("profile", {}).get("sha256") == loaded.manifest_sha256
                for o in row.get("outputs", [])
            )
        ):
            raise ProfileError("Native receipt, complete Job and Profile/build do not agree")
        for page in report["pages"]:
            candidate = page.get("candidate", {})
            if not (
                candidate.get("recipe_sha256") == recipe.sha256
                and candidate.get("semantic_identity_preserved") is True
                and candidate.get("parser_trace_preserved") is True
                and candidate.get("whole_recipe_contact_audit", {}).get("unreviewed_contact_count")
                == 0
            ):
                raise ProfileError("Incomplete recipe/semantic/contact validation")
        reports.append(platform.job.report.sha256)
    author = AuthorReview.model_validate(_record(evidence_root, request, request.author))
    if not (
        author.approved
        and author.profile_sha256 == loaded.manifest_sha256
        and set(author.job_report_sha256) == set(reports)
    ):
        raise ProfileError("Author review does not bind the exact qualified pages")
    if author.kind == "scoped-change-waiver":
        _validate_waiver(loaded, request, evidence_root, author, reports, pixels=pixels)
    mathematics = MathReview.model_validate(_record(evidence_root, request, request.math_review))
    if not (
        mathematics.independent_math_verified and set(mathematics.job_report_sha256) == set(reports)
    ):
        raise ProfileError("Independent mathematics review is missing or stale")
    return {
        "strategy": "personal-cpu-qualified",
        "profile_sha256": loaded.manifest_sha256,
        "recipe_sha256": recipe.sha256,
        "build_commit": current["commit"],
        "environment": environment.model_dump(),
        "scope": "local-release-qualification",
    }


def qualify(store, spec: str, evidence: Path) -> dict:
    """Validate and snapshot evidence under a writer lock; never publish or activate."""
    loaded = store.resolve(spec, candidate=True)
    with store.writer(loaded.manifest.profile):
        loaded = store.resolve(spec, candidate=True)
        request = Request.model_validate_json(evidence.read_bytes())
        result = _validate(loaded, request, evidence.parent, pixels=True)
        qid = digest(request.model_dump(mode="json", by_alias=True))
        destination = store.profile_dir(loaded.manifest.profile) / "qualifications" / qid
        if destination.exists():
            raise ProfileError("Qualification already exists; old evidence is immutable")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".qualify-", dir=destination.parent) as tmp:
            stage = Path(tmp) / "qualified"
            stage.mkdir()
            for name in request.files:
                target = inside(stage / "evidence", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(inside(evidence.parent, name), target)
            atomic_json(stage / "request.json", request.model_dump(mode="json", by_alias=True))
            _validate(loaded, request, stage / "evidence", pixels=False)
            receipt = Receipt(
                schema="personal-cpu-qualification/1",
                qualification_id=qid,
                profile=spec,
                profile_sha256=loaded.manifest_sha256,
                request_sha256=sha256(stage / "request.json"),
                created_at=datetime.now(UTC).isoformat(),
                status="locally-qualified",
            )
            atomic_json(stage / "receipt.json", receipt.model_dump(mode="json", by_alias=True))
            stage.rename(destination)
    return {**result, "qualification_id": qid, "published": False, "activated": False}


def _qualified(store, loaded, qid, *, pixels=False):
    if not re.fullmatch(r"[a-f0-9]{64}", qid):
        raise ProfileError("Invalid qualification identity")
    folder = inside(store.profile_dir(loaded.manifest.profile), f"qualifications/{qid}")
    receipt = Receipt.model_validate_json(inside(folder, "receipt.json").read_bytes())
    request_path = inside(folder, "request.json")
    request = Request.model_validate_json(request_path.read_bytes())
    if not (
        receipt.qualification_id == qid
        and receipt.profile == loaded.manifest.ref
        and receipt.profile_sha256 == loaded.manifest_sha256
        and receipt.request_sha256 == sha256(request_path)
        and digest(request.model_dump(mode="json", by_alias=True)) == qid
    ):
        raise ProfileError("Qualification receipt identity changed")
    result = _validate(loaded, request, folder / "evidence", pixels=pixels)
    return {**result, "qualification_id": qid, "receipt_sha256": sha256(folder / "receipt.json")}


def publish_ink(store, spec: str, qualification_id: str) -> dict:
    """Explicit local publication; a crash before its receipt leaves an inert snapshot."""
    candidate = store.resolve(spec, candidate=True)
    with store.writer(candidate.manifest.profile):
        candidate = store.resolve(spec, candidate=True)
        qualified = _qualified(store, candidate, qualification_id, pixels=True)
        folder = store.profile_dir(candidate.manifest.profile)
        record = folder / "publications" / f"{candidate.manifest.version}.json"
        if record.exists():
            raise ProfileError("Published qualification is immutable")
        destination = folder / "versions" / candidate.manifest.version
        if destination.exists():
            if store.resolve(spec).manifest_sha256 != candidate.manifest_sha256:
                raise ProfileError("Existing version differs from candidate; refusing overwrite")
        else:
            copy_snapshot(candidate, destination)
        publication = Publication(
            schema="personal-cpu-publication/1",
            profile=spec,
            profile_sha256=candidate.manifest_sha256,
            qualification_id=qualification_id,
            published_at=datetime.now(UTC).isoformat(),
        )
        atomic_json(record, publication.model_dump(mode="json", by_alias=True))
    return {"published": spec, "activated": False, "qualification": qualified}


def validate_published(store, loaded) -> dict:
    path = inside(
        store.profile_dir(loaded.manifest.profile), f"publications/{loaded.manifest.version}.json"
    )
    if not path.is_file():
        raise ProfileError(
            "Personal ink cannot become the default or render as font-cpu without "
            "local publication qualification; candidates require personal-cpu-experimental"
        )
    publication = Publication.model_validate_json(path.read_bytes())
    if (
        publication.profile != loaded.manifest.ref
        or publication.profile_sha256 != loaded.manifest_sha256
    ):
        raise ProfileError("Published qualification belongs to a different Profile")
    return _qualified(store, loaded, publication.qualification_id)


def activate_ink(store, spec: str) -> dict:
    loaded = store.resolve(spec)
    with store.writer(loaded.manifest.profile), lock(store.root / ".settings.lock"):
        loaded = store.resolve(spec)
        result = validate_published(store, loaded)
        settings = store.settings()
        settings["default_profile"] = loaded.manifest.ref
        settings.setdefault("active_versions", {})[loaded.manifest.profile] = (
            loaded.manifest.version
        )
        atomic_json(store.root / "settings.json", settings)
    return {"active": spec, "qualification": result}
