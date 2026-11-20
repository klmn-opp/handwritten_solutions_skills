"""Five uppercase-Phi body-reference probes, with both frozen source controls."""

import argparse
import copy
import shutil
from pathlib import Path

from PIL import Image
from pypdf import PdfWriter

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_delivery import crop_pair, json_value, snapshot_sources
from research.corpus.math_glyph_completion import local_comparison
from research.homework_benchmark.render import read, save_pdf
from research.math_feedback_candidate import paired_panel
from research.session_o_baseline import load_baseline, replay_page, validate_increment
from research.session_o_delivery import bank_inventory, verify_source_placements
from research.session_o_resolver import SessionOResolver

PRIOR = "20261120T134632687261Z-session-o-reviewed-math-complete-20-adf8f1c8"
PRIOR_SHA = "99b456d9e0de2dfb01f1ef4144082ae12eb55f60023fcc70a737c1c112f9b79c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    bank = args.bank.resolve()
    _, inventory = bank_inventory(bank)
    baseline = load_baseline()
    prior = workspace_root() / "runs" / PRIOR
    verify_run(prior)
    if sha256(prior / "run.json") != PRIOR_SHA:
        raise ValueError("Previous whole-box-fit control changed")
    run = Run.create(
        "session-o-phi-body-reference-probe",
        seed=1909,
        config={
            "baseline": str(baseline["path"]),
            "baseline_sha256": baseline["run_sha256"],
            "prior_whole_box_fit": PRIOR,
            "prior_sha256": PRIOR_SHA,
            "bank": str(bank),
            "bank_manifest_sha256": sha256(bank / "sources.json"),
            "pages": [11, 13],
            "expected_phi_count": 5,
            "targets": ["Φ"],
            "render": baseline["cfg"].model_dump(mode="json"),
            "selection_or_scale_search": False,
            "neural_updates": 0,
        },
    )
    print(run.path, flush=True)
    try:
        snapshot_sources(run, {"four-af-baseline": baseline["path"], "whole-box-fit": prior})
        shutil.copytree(bank, run.path / "solutions/glyphs")
        shutil.copytree(baseline["path"] / "solutions", run.path / "solutions/baseline-assets")
        atomic_json(run.path / "reports/bank-files.json", inventory)
        bank_copy = run.path / "solutions/glyphs"
        results, old_font_panels, old_personal_panels = [], [], []
        for number in (11, 13):
            replay = replay_page(baseline, number)
            case_id = replay["entry"]["case_id"]
            page, renderer = replay["page"], replay["renderer"]
            before = copy.deepcopy(replay["records"])
            resolver = SessionOResolver(bank_copy, seed=1909, targets=frozenset("Φ"))
            after = json_value(resolver.apply(page, renderer, before, case_id=case_id))
            image = save_pdf(
                renderer.to_vector_pdf([page]), run.path / f"rendered/pages/{number:02d}.pdf"
            )
            checks = validate_increment(
                replay["geometry"],
                page,
                renderer,
                before,
                after,
                replay["image"],
                image,
                targets={"Φ"},
                parser_before=replay["parser"],
            )
            sources = verify_source_placements(
                bank_copy, before, after, replay["geometry"], page, renderer
            )
            decisions = json_value(resolver.decisions)
            previous = read(prior / f"reports/{case_id}-provenance.json")["candidate"]
            previous_image = Image.open(prior / f"rendered/pages/{number:02d}.png").convert("RGB")
            ids = [
                i
                for i, row in enumerate(before)
                if row["context"] == "mathematics" and row["semantic"] == "Φ"
            ]
            if {d["item_index"] for d in decisions} != set(ids):
                raise ValueError("Not every intended Phi received an explicit decision")
            for i in ids:
                if after[i].get("session_o_source"):
                    if previous[i]["session_o_source"]["id"] != after[i]["session_o_source"]["id"]:
                        raise ValueError("Source identity changed between size controls")
                    if (
                        previous[i]["session_o_source"]["crop_sha256"]
                        != after[i]["session_o_source"]["crop_sha256"]
                    ):
                        raise ValueError("Source pixels changed between size controls")
                a, c = crop_pair(before[i], after[i], replay["image"], image, (75, 45))
                b, d = crop_pair(previous[i], after[i], previous_image, image, (75, 45))
                old_font_panels.append((number, i, a, c))
                old_personal_panels.append((number, i, b, d))
                paired_panel(
                    a,
                    c,
                    run.path / f"comparisons/old-font/{number:02d}-{i:04d}.png",
                    f"P{number} Phi U+03A6 / item {i}",
                    ("Original font", "Personal circle-body fit"),
                )
                paired_panel(
                    b,
                    d,
                    run.path / f"comparisons/old-personal/{number:02d}-{i:04d}.png",
                    f"P{number} same original Phi / item {i}",
                    ("Previous whole-box fit", "Circle-body fit"),
                )
            entry = {
                "page": number,
                "case_id": case_id,
                "decisions": decisions,
                "checks": checks,
                "source_placements": sources,
            }
            results.append(entry)
            atomic_json(
                run.path / f"reports/{case_id}.json",
                {
                    **entry,
                    "before": before,
                    "candidate": after,
                    "parser": replay["parser"],
                    "prior_whole_box_phi": [previous[i] for i in ids],
                },
            )
            print(
                number,
                "Phi accepted",
                sum(d["accepted"] for d in decisions),
                "/",
                len(ids),
                flush=True,
            )
        if sum(len(row["decisions"]) for row in results) != 5:
            raise ValueError("Expected exactly five frozen Phi instances")
        with PdfWriter() as writer:
            for number in (11, 13):
                writer.append(run.path / f"rendered/pages/{number:02d}.pdf")
            writer.write(run.path / "rendered/phi-probe-pages.pdf")
        for name, panels, labels in (
            ("old-font", old_font_panels, ("Original font", "Circle-body fit")),
            ("old-personal", old_personal_panels, ("Previous whole-box fit", "Circle-body fit")),
        ):
            output = run.path / "comparisons" / name
            local_comparison(output, panels, labels=labels)
        for name, digest in inventory.items():
            if sha256(bank / name) != digest or sha256(bank_copy / name) != digest:
                raise ValueError("Original bank changed")
        result = {
            "pages": [11, 13],
            "targets": 5,
            "accepted": sum(d["accepted"] for row in results for d in row["decisions"]),
            "same_source_id_and_rgb_as_whole_box_fit": True,
            "non_phi_records_geometry_and_target_external_pixels_exact": True,
            "all_source_transform_vertices_codes_exact": True,
            "parameter_search": False,
            "root_visual_review": "pending",
            "author_acceptance": False,
        }
        run.finish("held", result=result)
        verify_run(run.path)
        print(result, flush=True)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
