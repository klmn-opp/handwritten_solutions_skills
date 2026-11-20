"""Snapshot installed metadata and license file hashes; not an automatic legal approval."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = []
    seen = set()
    for dist in sorted(
        importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()
    ):
        meta = dist.metadata
        identity = (meta["Name"].lower().replace("_", "-"), dist.version)
        if identity in seen:
            continue
        seen.add(identity)
        licenses = []
        for file in dist.files or []:
            if any(word in str(file).lower() for word in ("license", "copying", "copyright")):
                path = dist.locate_file(file)
                if path.is_file():
                    licenses.append(
                        {"path": str(file), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    )
        packages.append(
            {
                "name": meta["Name"],
                "version": dist.version,
                "license_expression": meta.get("License-Expression"),
                "license_header": (meta.get("License") or "")[:200],
                "license_classifiers": [
                    v for v in meta.get_all("Classifier", []) if v.startswith("License")
                ],
                "requires": dist.requires or [],
                "license_files": licenses,
                "direct_url": json.loads(dist.read_text("direct_url.json") or "null"),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"packages": packages}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Recorded {len(packages)} distributions in {args.output}")


if __name__ == "__main__":
    main()
