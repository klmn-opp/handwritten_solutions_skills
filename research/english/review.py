"""Content-first blind review packet and strict ingestion of genuine reviewer answers."""

import argparse
import base64
import json
import random
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256

from .artifacts import create, lossless_pdf
from .metrics import aggregate, score

AXES = ["connections", "letter_proportions", "spacing_rhythm", "personal_similarity", "legibility"]


def validate_answers(packet, answers):
    if not answers.get("reviewer") or answers.get("reviewer_type") != "human":
        raise ValueError("Named human reviewer required; model judgments are not human ratings")
    if answers.get("packet_run_id") != packet["run_id"]:
        raise ValueError("Review packet mismatch")
    known = {r["blind_id"]: r for r in packet["items"]}
    seen = set()
    results = []
    for row in answers["answers"]:
        bid = row["blind_id"]
        if bid not in known or bid in seen:
            raise ValueError("Unknown or duplicate blind ID")
        seen.add(bid)
        if row.get("content") not in {"complete", "incorrect", "uncertain"}:
            raise ValueError("Explicit content adjudication required")
        transcript = row.get("transcription")
        if not isinstance(transcript, str):
            raise ValueError("Blind transcription required")
        ratings = row.get("style")
        if ratings is not None:
            if row["content"] != "complete":
                raise ValueError("Style requires verified complete content")
            if set(ratings) != set(AXES) or any(
                type(v) is not int or not 1 <= v <= 5 for v in ratings.values()
            ):
                raise ValueError("All five style dimensions require integers 1..5")
        if (
            row["content"] == "complete"
            and not score(known[bid]["expected"], transcript)["exact_match"]
        ):
            raise ValueError(
                "Complete content disagrees with blind transcription; adjudicate separately"
            )
        results.append(
            {**row, "human_transcription_score": score(known[bid]["expected"], transcript)}
        )
    return {
        "reviewer": answers["reviewer"],
        "reviewer_type": "human",
        "items": results,
        "remaining": len(known) - len(seen),
        "transcription_metrics": aggregate(
            [r["human_transcription_score"] for r in results if r["content"] != "uncertain"]
        ),
        "interpretation": "Single human review; independent second adjudication still recommended",
    }


def packet(args):
    data = json.loads((args.evaluation / "reports/observations.json").read_text())
    selected = [
        r for r in data["rows"] if r.get("smoke") or r["category"] == "real-personal-calibration"
    ]
    # Same samples, separate raw/fit variants remain distinct blinded candidates.
    random.Random(805).shuffle(selected)
    run = create(
        "english-blind-review",
        run_id=args.run_id,
        seed=805,
        config={
            "evaluation": str(args.evaluation),
            "evaluation_sha256": sha256(args.evaluation / "run.json"),
            "content_before_style": True,
            "human_answers_present": False,
        },
    )
    public = []
    private = []
    images = []
    for i, row in enumerate(selected):
        bid = f"B{i + 1:03d}"
        source = args.evaluation / "rendered/pages" / (row["evaluation_id"] + ".png")
        path = run.path / "rendered/pages" / f"{bid}.png"
        path.write_bytes(source.read_bytes())
        images.append(path)
        public.append(
            {
                "blind_id": bid,
                "expected": row["text"],
                "image": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(),
            }
        )
        private.append(
            {
                "blind_id": bid,
                "parent_run": row["parent_run"],
                "case_id": row["id"],
                "seed": row.get("seed"),
                "profile": row.get("profile"),
                "image_sha256": sha256(path),
            }
        )
    author = [r for r in data["rows"] if r["category"] == "real-personal-calibration"][:2]
    reference_images = [
        "data:image/png;base64," + base64.b64encode(Path(r["original_image"]).read_bytes()).decode()
        for r in author
    ]
    payload = {"run_id": run.path.name, "items": public, "author_references": reference_images}
    atomic_json(run.path / "reports/reviewer-packet.json", payload)
    atomic_json(run.path / "reports/private-unblinding-key.json", private)
    (run.path / "comparisons/blind-review.html").write_text(
        Path(__file__)
        .with_suffix(".html")
        .read_text()
        .replace("PACKET", json.dumps(payload).replace("<", "\\u003c"))
    )
    atomic_json(
        run.path / "reports/human-answers-template.json",
        {"packet_run_id": run.path.name, "reviewer": "", "reviewer_type": "human", "answers": []},
    )
    atomic_json(
        run.path / "reports/pdf-integrity.json",
        lossless_pdf(images, run.path / "rendered/homework_solutions.pdf"),
    )
    print(
        run.finish(
            "held", result={"blind_items": len(public), "human_ratings": 0, "content_verified": 0}
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("packet")
    a.add_argument("--evaluation", type=Path, required=True)
    a.add_argument("--run-id", required=True)
    a = sub.add_parser("ingest")
    a.add_argument("--packet", type=Path, required=True)
    a.add_argument("--answers", type=Path, required=True)
    a.add_argument("--run-id", required=True)
    args = p.parse_args()
    if args.command == "packet":
        packet(args)
    else:
        result = validate_answers(
            json.loads(args.packet.read_text()), json.loads(args.answers.read_text())
        )
        run = create(
            "english-human-adjudication",
            run_id=args.run_id,
            config={"packet_sha256": sha256(args.packet), "answers_sha256": sha256(args.answers)},
        )
        atomic_json(run.path / "reports/adjudication.json", result)
        print(
            run.finish(
                "held",
                result={"human_reviews": len(result["items"]), "publication_approved": False},
            )
        )


if __name__ == "__main__":
    main()
