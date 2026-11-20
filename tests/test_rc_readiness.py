"""Synthetic receipt-contract tests, not native-platform execution evidence."""

import hashlib
import json

from scripts.rc_readiness import assess


def test_missing_platforms_and_author_cannot_qualify(tmp_path):
    result = assess({"candidate": {"version": "1.0.0rc1"}}, tmp_path)
    assert result["decision"] == "No-Go"
    assert len(result["reasons"]) == 4
    assert result["production_activation"] is False


def test_linux_label_cannot_stand_in_for_mac(tmp_path):
    candidate = dict(
        version="1.0.0rc1",
        source_commit="a" * 40,
        wheel_sha256="b" * 64,
        profile_archive_sha256="c" * 64,
        job_sha256="d" * 64,
    )
    row = dict(
        schema="cross-platform-candidate-check/1",
        actual_system="Linux",
        status="passed",
        clean_install_completed=True,
        installed_wheel_members_exact=True,
        source_commit=candidate["source_commit"],
        package_version=candidate["version"],
        inputs={
            key: {"sha256": candidate[field]}
            for key, field in [
                ("wheel", "wheel_sha256"),
                ("profile", "profile_archive_sha256"),
                ("job", "job_sha256"),
            ]
        },
    )
    path = tmp_path / "linux.json"
    path.write_text(json.dumps(row))
    entry = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result = assess(
        {"candidate": candidate, "platforms": {"Linux": entry, "Darwin": entry}}, tmp_path
    )
    assert result["decision"] == "No-Go"
    assert result["platform_checks"][0]["accepted"]
    assert not result["platform_checks"][1]["accepted"]
    path.write_text("{}")
    result = assess({"candidate": candidate, "platforms": {"Linux": entry}}, tmp_path)
    assert "evidence absent or changed" in result["reasons"][0]
