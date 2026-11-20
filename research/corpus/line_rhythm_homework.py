"""Relax tight English word gaps with fixed formulas and unchanged ink patches."""

import shutil
from pathlib import Path

import numpy as np

from research.corpus.context_content_train import digest, write
from research.corpus.context_metric_evaluate import read


def adjacent(left: dict, right: dict, a: np.ndarray, b: np.ndarray) -> bool:
    return bool(
        left["items"][-1] + 1 == right["items"][0]
        and min(a[3], b[3]) > max(a[1], b[1])
        and b[0] >= a[2]
    )


def relax(
    words: list[dict],
    occupied: np.ndarray,
    cores: dict,
    target: float,
    page_size: tuple,
    right_limit: float,
) -> tuple[dict, list[dict]]:
    from research.corpus.mixed_ink_pages import collision_reason

    runs = []
    active = []
    for word in words:
        key = word["word_id"]
        if key not in cores:
            if active:
                runs.append(active)
                active = []
            continue
        if active and not adjacent(active[-1], word, cores[active[-1]["word_id"]], cores[key]):
            runs.append(active)
            active = []
        active.append(word)
    if active:
        runs.append(active)
    shifts, measurements = {}, []
    for segment in runs:
        if len(segment) < 2:
            continue
        boxes = np.asarray([cores[w["word_id"]] for w in segment])
        heights = boxes[:, 3] - boxes[:, 1]
        local_height = (heights[1:] + heights[:-1]) / 2
        gaps = boxes[1:, 0] - boxes[:-1, 2]
        extra = np.minimum(np.maximum(target * local_height - gaps, 0) / 2, 0.15 * local_height)
        requested = np.r_[0, np.cumsum(np.rint(extra).astype(int))]
        ids = [i for word in segment for i in word["items"]]
        chosen = np.zeros(len(segment), dtype=int)
        for divisor in [1, 2, 4, 8, 16]:
            proposal = requested // divisor
            moves = [
                occupied[word["items"][0]] + [dx, 0, dx, 0]
                for word, dx in zip(segment, proposal, strict=True)
            ]
            if all(
                box[2] <= right_limit and collision_reason(box, occupied, ids, *page_size) is None
                for box in moves
            ):
                chosen = proposal
                break
        for word, shift in zip(segment, chosen, strict=True):
            if shift:
                shifts[word["word_id"]] = int(shift)
                occupied[word["items"]] += [shift, 0, shift, 0]
        revised = gaps + np.diff(chosen)
        for left, right, before, after, unit in zip(
            segment[:-1], segment[1:], gaps, revised, local_height, strict=True
        ):
            measurements.append(
                {
                    "left_word": left["word_id"],
                    "right_word": right["word_id"],
                    "before": float(before / unit),
                    "after": float(after / unit),
                }
            )
    return shifts, measurements


def launch(mixed: Path, registry: Path, fit: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create
    from research.corpus.metric_mixed_homework import assemble

    for p in [mixed, registry, fit]:
        verify_run(p)
    assert not read(fit / "reports/result.json")["passed_context_gap_gate"]
    source = Path(read(mixed / "reports/request.json")["source"])
    verify_run(source)
    cfg = {
        "mixed": str(mixed),
        "mixed_sha256": digest(mixed / "run.json"),
        "source": str(source),
        "source_sha256": digest(source / "run.json"),
        "registry": str(registry),
        "registry_sha256": digest(registry / "run.json"),
        "fit": str(fit),
        "fit_sha256": digest(fit / "run.json"),
        "target": read(registry / "reports/result.json")["gap_normalized_quartiles"][0],
        "C_used": False,
        "new_neural_updates": 0,
    }
    run = create("measured-word-rhythm-mixed-homework", seed=2012, config=cfg)
    write(run.path / "reports/request.json", cfg)
    Path("/tmp/context-v01023-rhythm-homework").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        for md in (mixed / "solutions").glob("problem-*.md"):
            shutil.copyfile(md, run.path / "solutions" / md.name)
        shutil.copyfile(
            mixed / "rendered/homework_solutions.pdf", run.path / "comparisons/v22-homework.pdf"
        )
        result = assemble(
            run.path,
            source,
            word_manifest=mixed / "reports/neural-words.json",
            rhythm_gap=cfg["target"],
        )
        prior = read(mixed / "reports/page-provenance.json")
        revised = read(run.path / "reports/page-provenance.json")
        gaps = []
        for old, new in zip(prior, revised, strict=True):
            assert old["math_glyph_provenance"] == new["math_glyph_provenance"]
            for a, b in zip(old["word_provenance"], new["word_provenance"], strict=True):
                assert a.get("sha256") == b.get("sha256")
                assert a["fallback_reason"] == b["fallback_reason"]
                assert a["items"] == b["items"]
            gaps.extend(new["word_rhythm"])
        result.update(
            {
                "changed_word_placements": sum(q["changed_word_placements"] for q in revised),
                "pair_count": len(gaps),
                "before_median_gap": float(np.median([q["before"] for q in gaps])),
                "after_median_gap": float(np.median([q["after"] for q in gaps])),
                "ink_asset_hashes_and_fallbacks_equal": True,
            }
        )
        run.finish("held", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path
