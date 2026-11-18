"""Repeatable CLI lifecycle demonstration with explicitly synthetic stroke samples."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "HWSOL_DATA_DIR": str(root / "studio-data")}
    history = []

    def run(*arguments, target_env=None):
        result = subprocess.run(
            [sys.executable, "-m", "hwsol", *map(str, arguments)],
            env=target_env or env,
            capture_output=True,
            text=True,
            check=True,
        )
        value = json.loads(result.stdout)
        history.append({"command": list(map(str, arguments)), "result": value})
        return value

    run("doctor")
    run("profile", "template", "--output", root / "template.pdf", "--chars", "xA")
    source = root / "synthetic.md"
    source.write_text(
        "# Synthetic Profile fixture\n\n"
        "Engineering strokes, not an author's handwriting or a homework answer.\n\n"
        "xxxxxxxxxxxx xxxxxxxxxxxx\n\n$$x^2=4,\\quad x=2$$\n",
        encoding="utf-8",
    )
    run("render-probe", "--input", source, "--output", root / "bundled")
    for n in (1, 2):
        sample = root / f"round-{n}.json"
        sample.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source": "synthetic-engineering-fixture",
                    "cells": [
                        {
                            "index": 0,
                            "strokes": [
                                [[0.2, 0.2], [0.8, 0.8]],
                                [[0.2, 0.8], [0.8, 0.1 * n]],
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        run(
            "profile",
            "import",
            "--profile",
            "demo",
            "--source",
            sample,
            "--template",
            root / "template.json",
            "--license-label",
            "CC0-1.0-synthetic",
        )
        run("profile", "train", "--profile", "demo", "--append")
        run("profile", "evaluate", "--profile", "demo", "--version", f"v{n}")
        run("profile", "publish", "--profile", "demo", "--version", f"v{n}")
        run("profiles", "activate", f"demo@v{n}")
        run("render-probe", "--input", source, "--output", root / f"v{n}")
    run("profiles", "compare", "demo@v1", "demo@v2")
    run("profiles", "rollback", "demo@v1")
    run("render-probe", "--input", source, "--output", root / "rollback")
    run("profiles", "export", "demo@v2", "--output", root / "demo-v2.zip")
    other = {**env, "HWSOL_DATA_DIR": str(root / "runtime-data")}
    run("profiles", "import", "--source", root / "demo-v2.zip", target_env=other)
    run(
        "render-probe",
        "--input",
        source,
        "--output",
        root / "portable",
        "--profile",
        "demo@v2",
        target_env=other,
    )
    reports = {
        label: json.loads((root / label / "validation_report.json").read_text("utf-8"))
        for label in ("bundled", "v1", "v2", "rollback", "portable")
    }
    assert reports["v1"]["artifacts"] == reports["rollback"]["artifacts"]
    assert reports["v2"]["artifacts"] == reports["portable"]["artifacts"]
    summary = {
        "status": "passed",
        "phase": 1,
        "sample_kind": "synthetic-strokes",
        "real_author_handwriting_validated": False,
        "rollback_byte_identical": True,
        "portable_byte_identical": True,
        "commands": len(history),
        "renders": {
            label: {key: r[key] for key in ("profile", "artifacts", "page_count")}
            for label, r in reports.items()
        },
    }
    (root / "commands.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
