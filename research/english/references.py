"""Reuse historical scan annotations, never infer real online trajectories."""

import json
from pathlib import Path

from PIL import Image

from hwsol.profiles.io import atomic_json, sha256

from .artifacts import create, lossless_pdf


def prepare(run_id):
    source = Path("samples/realism-inbox/round-2/layout-english.jpg")
    annotations = Path("profiles/realism-studio/samples/personal")
    regions = annotations / "s-74ce14014b764820951e21955825596f/original-002.json"
    geometry = json.loads(regions.read_text())
    assert sha256(source) == geometry["english-upper"]["source_sha256"]
    run = create(
        "english-personal-reference",
        run_id=run_id,
        seed=42,
        config={
            "source": str(source),
            "source_sha256": sha256(source),
            "annotation_sha256": sha256(regions),
            "no_training": True,
        },
    )
    rows = []
    with Image.open(source) as scan:
        for section, sample in [
            ("english-upper", "s-095108b2487f49c08391fca51701f48c"),
            ("english-lower", "s-b9eca32cad5443bda196bfc31d04e382"),
        ]:
            annotation = annotations / sample / "original-000.json"
            lines = json.loads(annotation.read_text())["lines"]
            group = geometry[section]
            view = scan.crop(group["view"])
            for i, (region, line) in enumerate(zip(group["regions"], lines, strict=True)):
                sid = f"personal-{section}-{i + 1:02d}"
                dest = run.path / "rendered/pages" / f"{sid}.png"
                view.crop(region["box"]).save(dest)
                rows.append(
                    {
                        "id": sid,
                        "category": "real-personal-calibration",
                        "split": "calibration",
                        "text": line["text"],
                        "status": "generated",
                        "backend": "real-personal-scan",
                        "image": str(dest.relative_to(run.path)),
                        "image_sha256": sha256(dest),
                        "annotation_source": str(annotation),
                        "annotation_sha256": sha256(annotation),
                        "human_reverified": False,
                        "source_type": "scanned-ink-contour",
                        "online_trajectory": False,
                    }
                )
            if section == "english-upper":
                first = lines[0]
                box = group["regions"][0]["box"]
                selected = [0, 2, 3, 6, 9]
                wordrefs = []
                for idx in selected:
                    left, right = first["words"][idx]
                    crop = view.crop((max(0, int(left) - 4), box[1], int(right) + 4, box[3]))
                    dst = run.path / "comparisons" / f"word-{idx}.png"
                    crop.save(dst)
                    wordrefs.append(
                        {
                            "image": str(dst.resolve()),
                            "sha256": sha256(dst),
                            "text": first["text"].split()[idx],
                            "source_box": [left, box[1], right, box[3]],
                            "source_type": "real-scan",
                        }
                    )
                atomic_json(run.path / "reports/five-word-references.json", wordrefs)
    atomic_json(
        run.path / "reports/lines.json",
        {
            "cases": rows,
            "annotation_status": "historical transcription, pending independent recheck",
        },
    )
    atomic_json(
        run.path / "reports/pdf-integrity.json",
        lossless_pdf(
            [run.path / r["image"] for r in rows], run.path / "rendered/homework_solutions.pdf"
        ),
    )
    print(
        run.finish(
            "held", result={"lines": len(rows), "real_scan": True, "human_reverified": False}
        )
    )


if __name__ == "__main__":
    import sys

    prepare(sys.argv[1])
