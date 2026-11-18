"""Prepare seven historical real scans as review candidates, never verified labels."""

import json
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from research.corpus.experiment import create
from research.corpus.ingest import import_file, propose_lines
from research.corpus.store import Corpus


def prepare(corpus: Corpus, run_id: str) -> Path:
    run = create(
        "corpus-historical-scan-pilot",
        run_id=run_id,
        config={"corpus": str(corpus.root), "human_verified": False},
    )
    try:
        files = sorted(Path("samples/realism-inbox").glob("*.jpg")) + sorted(
            Path("samples/realism-inbox/round-2").glob("*.jpg")
        )
        docs = [
            import_file(
                corpus,
                p,
                course="historical collection sheets",
                batch="legacy-calibration-exposed",
                writer="personal",
            )
            for p in files
        ]
        lookup = {p["source_hash"]: p for p in corpus.rows("page")}
        base = Path("profiles/realism-studio/samples/personal")
        geo = base / "s-74ce14014b764820951e21955825596f/original-002.json"
        geometry = json.loads(geo.read_text())
        rows = []
        for section, sid in [
            ("english-upper", "s-095108b2487f49c08391fca51701f48c"),
            ("english-lower", "s-b9eca32cad5443bda196bfc31d04e382"),
            ("math-left", "s-2930efd3f59f4ed481dfbd0d48939c2c"),
            ("math-right", "s-0920d8731e3343f39bbbe29dd0bc02b4"),
        ]:
            group = geometry[section]
            page = lookup[group["source_sha256"]]
            labels = json.loads((base / sid / "original-000.json").read_text())["lines"]
            for region, label in zip(group["regions"], labels, strict=True):
                x0, y0, x1, y1 = region["box"]
                vx, vy = group["view"][:2]
                sample = corpus.propose(
                    page["page_id"],
                    [x0 + vx, y0 + vy, x1 + vx, y1 + vy],
                    content_type="english-line"
                    if section.startswith("english")
                    else "math-expression",
                    label=label.get("text", ""),
                    label_source="legacy-calibration-pending-human-recheck",
                    role="personal-handwriting",
                    notes=(
                        f"Source annotation {geo}; SHA256 {sha256(geo)}; "
                        "historical calibration is not independent holdout."
                    ),
                )
                rows.append(sample)
        annotated = {s["page_id"] for s in rows}
        for page in lookup.values():
            if page["page_id"] not in annotated:
                rows.extend(propose_lines(corpus, page["page_id"]))
        atomic_json(run.path / "reports/imports.json", docs)
        atomic_json(run.path / "reports/candidates.json", rows)
        atomic_json(
            run.path / "reports/recognition-input.json",
            {
                "images": [
                    {
                        "id": s["sample_id"],
                        "image": str(corpus.path(s["image_crop"])),
                        "sha256": s["crop_sha256"],
                        "content_type": s["content_type"],
                    }
                    for s in rows
                ]
            },
        )
        run.finish(
            "held",
            result={
                "real_original_images": len(files),
                "candidate_regions": len(rows),
                "verified_labels": 0,
                "new_homework_pdfs": 0,
                "reason": "Requires human transcription and region review",
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise
