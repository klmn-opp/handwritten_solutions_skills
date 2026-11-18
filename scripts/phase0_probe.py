"""Run repeatable backend probes. No downloads and no real personal samples."""

import argparse
import json
from pathlib import Path

from matplotlib import font_manager

from hwsol.adapters.base import RenderError
from hwsol.adapters.handnote import HandnoteAdapter
from hwsol.adapters.markdown_to_handwrite import MarkdownToHandwriteAdapter
from hwsol.config import RenderConfig, test_font
from hwsol.probe import run_probe, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mth-python", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    common = (root / "examples/phase0/common.md").read_text("utf-8")
    cases = {
        "common": common,
        "multipage": "\n\n".join([common] * 3),
        "unsupported": r"$$\definitelyUnsupported{x}$$",
        "aligned": r"$$\begin{aligned}2x+3&=7\\2x&=4\\x&=2\end{aligned}$$",
        "missing-glyph": "Unsupported text: 汉",
        "empty": "",
    }
    adapters = [HandnoteAdapter()]
    if args.mth_python:
        adapters.append(MarkdownToHandwriteAdapter(args.mth_python))
    reports = []
    for adapter in adapters:
        for name, text in cases.items():
            source = target / f"{name}.md"
            source.write_text(text, encoding="utf-8")
            folder = target / f"{adapter.name}-{name}"
            try:
                report = run_probe(source, folder, test_font(), RenderConfig(), adapter)
            except RenderError:
                report = json.loads((folder / "validation_report.json").read_text("utf-8"))
            expected = "failed" if name in {"unsupported", "missing-glyph", "empty"} else "passed"
            if name == "aligned" and adapter.name == "handnote":
                expected = "failed"
            matched = report["status"] == expected
            if name == "multipage":
                matched = matched and report.get("page_count", 0) > 1
            reports.append(
                {
                    "case": name,
                    "backend": adapter.name,
                    "status": report["status"],
                    "expected_status": expected,
                    "expectation_met": matched,
                    "page_count": report.get("page_count"),
                    "elapsed_seconds": report["elapsed_seconds"],
                    "error": report.get("error"),
                    "report": str(folder.relative_to(target) / "validation_report.json"),
                }
            )
    comparison = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    for name, font, cfg in (
        ("font-switch", comparison, RenderConfig()),
        ("300dpi", test_font(), RenderConfig(dpi=300)),
    ):
        report = run_probe(
            target / "common.md", target / f"handnote-{name}", font, cfg, HandnoteAdapter()
        )
        reports.append(
            {
                "case": name,
                "backend": "handnote",
                "status": report["status"],
                "expected_status": "passed",
                "expectation_met": report["status"] == "passed",
                "page_count": report["page_count"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    passed = all(item["expectation_met"] for item in reports)
    write_json(
        target / "summary.json",
        {
            "scope": "phase0, synthetic fixtures",
            "probes": reports,
            "expected_behavior_verified": passed,
        },
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
