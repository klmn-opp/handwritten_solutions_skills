"""Check evidence bindings for a proposed RC; never publish or activate assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linked_record(base: Path, evidence: dict):
    path = base / evidence["path"]
    if not path.is_file() or digest(path) != evidence["sha256"]:
        raise ValueError("Installation evidence absent or changed")
    return json.loads(path.read_text())


def clean_installation(row: dict, evidence: dict, candidate: dict, base: Path) -> bool:
    """A new Profile test may reuse the exact previously clean-installed package.

    Installation proof binds the package and interpreter; current acceptance still
    separately binds the new Profile and Job. Never relabel a reuse receipt clean.
    """
    if row.get("clean_install_completed") is True:
        return True
    link = evidence.get("installation")
    if not link:
        return False
    try:
        parent = linked_record(base, link)
        external = None
        if isinstance(parent, list):
            selected = [p for p in parent if p.get("python") == link.get("python")]
            if len(selected) != 1:
                return False
            external = selected[0]
            if not (
                external.get("status") == "passed"
                and external.get("clean_install_completed") is True
                and external.get("source_commit") == candidate["source_commit"]
                and external.get("package_version") == candidate["version"]
                and external.get("wheel_sha256") == candidate["wheel_sha256"]
            ):
                return False
            parent = linked_record(
                (base / link["path"]).parent,
                {
                    "path": external["portable_receipt"],
                    "sha256": external["portable_receipt_sha256"],
                },
            )
            if not Path(parent["runtime"]["hwsol"]).is_relative_to(
                Path(external["fresh_environment"])
            ):
                return False
        return bool(
            parent.get("schema") == "cross-platform-candidate-check/1"
            and parent.get("status") == "passed"
            and parent.get("actual_system") == row.get("actual_system")
            and (external is not None or parent.get("clean_install_completed") is True)
            and parent.get("source_commit") == candidate["source_commit"]
            and parent.get("package_version") == candidate["version"]
            and parent.get("inputs", {}).get("wheel", {}).get("sha256") == candidate["wheel_sha256"]
            and parent.get("runtime", {}).get("hwsol")
            and parent["runtime"] == row.get("runtime")
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def assess(spec: dict, base: Path) -> dict:
    reasons = []
    checked = []
    candidate = spec["candidate"]
    for system in ("Linux", "Darwin"):
        evidence = spec.get("platforms", {}).get(system)
        if not evidence:
            reasons.append(f"{system}: native installation evidence missing")
            continue
        path = base / evidence["path"]
        if not path.is_file() or digest(path) != evidence["sha256"]:
            reasons.append(f"{system}: evidence absent or changed")
            continue
        row = json.loads(path.read_text())
        verified_members = row.get("installed_wheel_members_exact")
        members_checked = verified_members is True or (
            type(verified_members) is int and verified_members > 0
        )
        valid = (
            row.get("schema") == "cross-platform-candidate-check/1"
            and row.get("actual_system") == system
            and row.get("status") == "passed"
            and clean_installation(row, evidence, candidate, base)
            and members_checked
            and row.get("source_commit") == candidate["source_commit"]
            and row.get("package_version") == candidate["version"]
            and row.get("inputs", {}).get("wheel", {}).get("sha256") == candidate["wheel_sha256"]
            and row.get("inputs", {}).get("profile", {}).get("sha256")
            == candidate["profile_archive_sha256"]
            and row.get("inputs", {}).get("job", {}).get("sha256") == candidate["job_sha256"]
        )
        if not valid:
            reasons.append(
                f"{system}: native clean execution or exact candidate binding not satisfied"
            )
        checked.append({"system": system, "receipt_sha256": digest(path), "accepted": valid})
    author = spec.get("author_review")
    e2e = spec.get("end_to_end")
    if not e2e:
        reasons.append("Complete new-question intake/solution/verification evidence missing")
    else:
        path = base / e2e["path"]
        if not path.is_file() or digest(path) != e2e["sha256"]:
            reasons.append("End-to-end evidence absent or changed")
        else:
            row = json.loads(path.read_text())
            if not (
                row.get("full_job_passed") is True
                and row.get("independent_math_verified") is True
                and row.get("candidate") == candidate
                and row.get("original_questions_sha256")
                and row.get("reading_receipt_sha256")
            ):
                reasons.append("Complete new-content evidence does not bind this candidate")
    if not author:
        reasons.append("Author review of the exact full-page candidate missing")
    else:
        path = base / author["path"]
        if not path.is_file() or digest(path) != author["sha256"]:
            reasons.append("Author review receipt absent or changed")
        else:
            row = json.loads(path.read_text())
            if not (row.get("author_approved") is True and row.get("candidate") == candidate):
                reasons.append("Author approval does not bind this exact candidate")
    if not re.fullmatch(r"1\.0\.0rc[1-9][0-9]*", candidate["version"]):
        reasons.append("Package has no unique v1.0 RC metadata version")
    return {
        "schema": "rc-readiness/1",
        "candidate": candidate,
        "decision": "No-Go" if reasons else "eligible-for-release-review",
        "reasons": reasons,
        "platform_checks": checked,
        "production_activation": False,
        "scope": "Local evidence consistency gate; receipt truth requires independent review.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(json.loads(args.spec.read_text()), args.spec.parent)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["decision"] == "eligible-for-release-review" else 2)


if __name__ == "__main__":
    main()
