"""Queue missing symbols in Studio without interrupting ordinary rendering."""

import math
from collections import Counter
from pathlib import Path

from hwsol.config import RenderConfig
from hwsol.jobs.layout import render_one
from hwsol.jobs.schema import Background
from hwsol.profiles.io import atomic_json, sha256
from hwsol.training.studio import template


def collection_plan(profile, sources: list[Path], output: Path, *, sets: int = 4) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    counts, maths, structures = Counter(), Counter(), Counter()
    analyzed, failures, seen = [], [], set()
    for path in sources:
        fingerprint = sha256(path)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        try:
            _, report = render_one(
                path.read_text("utf-8"), "1", profile, RenderConfig(), Background(), None
            )
            for row in report["glyph_provenance"][1:]:  # Exclude the question label.
                if row.get("codepoint"):
                    cp = int(row["codepoint"][2:], 16)
                    counts[cp] += 1
                    if row.get("latex") or row.get("structure"):
                        maths[cp] += 1
                if row.get("structure"):
                    structures[row["structure"]] += 1
            analyzed.append({"path": str(path), "sha256": fingerprint})
        except (ValueError, OSError) as exc:
            failures.append({"path": str(path), "sha256": fingerprint, "error": str(exc)})
    personal = set(profile.manifest.personal_codepoints)
    rows = []
    for cp, count in counts.items():
        variants = profile.manifest.variants.get(str(cp), 1) if cp in personal else 0
        priority = (1 + math.log1p(count)) * (3 if maths[cp] else 1) / (variants + 1)
        rows.append(
            {
                "character": chr(cp),
                "codepoint": f"U+{cp:04X}",
                "frequency": count,
                "math_frequency": maths[cp],
                "personal_variants": variants,
                "needed_variants": max(0, sets - variants),
                "priority": priority,
                "reason": "missing-personal" if not variants else "variant-top-up",
            }
        )
    rows.sort(key=lambda row: (-row["priority"], row["codepoint"]))
    requested = [r for r in rows if r["needed_variants"] and r["character"].isprintable()]
    templates = []
    for start in range(0, len(requested), 64):
        chars = "".join(row["character"] for row in requested[start : start + 64])
        templates.append(template(output / f"symbols-{start // 64 + 1:02d}.pdf", chars, sets))
    report = {
        "schema_version": 1,
        "profile": profile.provenance(),
        "sources": analyzed,
        "source_failures": failures,
        "deduplicated_by": "content-sha256",
        "glyphs": rows,
        "templates": templates,
        "structure_tasks": [
            {
                "structure": key,
                "frequency": value,
                "capture": "annotated scan envelope or recorded strokes; keep source kind",
            }
            for key, value in sorted(structures.items())
        ],
        "runtime_blocking": False,
        "collection_backend": "locked-handnote-template-and-font-builder",
    }
    atomic_json(output / "collection-plan.json", report)
    return report
