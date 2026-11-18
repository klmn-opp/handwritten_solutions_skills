"""Fail-closed checkpoint/data lineage checks without importing Torch into CPU runtime."""

import json
from pathlib import Path

from hwsol.profiles.io import sha256


def verify_continuation(request: dict, checkpoint: dict) -> None:
    if checkpoint["labels"] != request["labels"]:
        raise ValueError("Continuation character identities changed")
    if checkpoint["arrays_sha256"] == request["arrays_sha256"]:
        return
    if not request.get("allow_data_extension"):
        raise ValueError("Continuation data changed without explicit extension")
    data = Path(request["data"])
    if sha256(data / "manifest.json") != request["manifest_sha256"]:
        raise ValueError("Extension manifest changed")
    manifest = json.loads((data / "manifest.json").read_text())
    certificate = manifest.get("extension", {})
    if (
        certificate.get("schema") != "append-only-reference-data/1"
        or certificate.get("parent_manifest_sha256") != checkpoint["dataset_manifest_sha256"]
        or certificate.get("parent_arrays_sha256") != checkpoint["arrays_sha256"]
        or certificate.get("new_arrays_sha256") != request["arrays_sha256"]
        or not certificate.get("old_samples_retained")
        or not certificate.get("reference_unchanged")
    ):
        raise ValueError("Checkpoint is not the certified parent dataset")
    parent = Path(certificate["parent_data"])
    if (
        sha256(parent / "manifest.json") != certificate["parent_manifest_sha256"]
        or sha256(parent / "arrays.npz") != certificate["parent_arrays_sha256"]
    ):
        raise ValueError("Parent dataset changed")
    if sha256(data / "extension.json") != manifest["files"]["extension.json"]:
        raise ValueError("Extension evidence changed")
