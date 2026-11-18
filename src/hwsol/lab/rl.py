"""CPU gradient-bandit experiments over immutable, measured renderer rollouts."""

import csv
import hashlib
import itertools
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from hwsol.jobs.runtime import load_job, render_job, verify_job
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.store import ProfileStore
from hwsol.render_policy import RenderPolicy, StrokeAction

ACTIONS = (
    StrokeAction(name="accepted-b", legacy=True),
    StrokeAction(name="balanced", ink_balance=3, continuity=1, repeat_penalty=0.8),
    StrokeAction(
        name="varied",
        ink_balance=1.5,
        continuity=0.5,
        repeat_penalty=2,
        lean=0.06,
        lean_variation=0.035,
    ),
    StrokeAction(
        name="steady",
        ink_balance=6,
        continuity=2,
        repeat_penalty=0.6,
        lean=0.075,
        lean_variation=0.012,
    ),
    StrokeAction(
        name="relaxed",
        ink_balance=2.5,
        continuity=1.2,
        repeat_penalty=1.5,
        lean=0.10,
        lean_variation=0.03,
    ),
    StrokeAction(
        name="light-varied",
        ink_balance=4.5,
        continuity=0.5,
        repeat_penalty=1.8,
        lean=0.04,
        lean_variation=0.025,
    ),
)
PRIOR = (0.25, 0.15, 0.15, 0.15, 0.15, 0.15)


def _renderer_stamp() -> dict:
    source = Path(__file__).resolve().parents[1]
    return {
        name: sha256(source / name)
        for name in (
            "config.py",
            "render_policy.py",
            "jobs/ink_policy.py",
            "jobs/handwriting.py",
            "jobs/layout.py",
            "adapters/profile_renderer.py",
            "jobs/math_strokes.py",
        )
    }


def proxy_reward(metrics: dict) -> float:
    """Engineering objective only: ink consistency, continuity, real-variant entropy."""
    return float(
        0.55 * np.exp(-2 * metrics["ink_log_rmse"])
        + 0.25 * np.exp(-2 * metrics["ink_step"])
        + 0.20 * metrics["variant_entropy"]
    )


def gradient_bandit(
    rewards: list[list[float]], prior: list[float], *, seed: int, steps: int = 800
) -> dict:
    """REINFORCE for a one-step environment, with a running baseline and KL/entropy."""
    if len(rewards) != len(prior) or not all(rewards):
        raise ValueError("Every action needs at least one reward")
    if not 20 <= steps <= 20000:
        raise ValueError("Choose 20..20000 bandit steps")
    flat = np.concatenate([np.asarray(row, dtype=float) for row in rewards])
    if not np.isfinite(flat).all() or flat.min() < 0 or flat.max() > 1:
        raise ValueError("Rewards must be finite values in [0, 1]")
    prior = np.asarray(prior, dtype=float)
    if not np.isfinite(prior).all() or np.any(prior <= 0) or abs(prior.sum() - 1) > 1e-8:
        raise ValueError("Bandit prior must have positive probabilities summing to one")
    rng = np.random.default_rng(seed)
    logits = np.log(prior)
    baseline = float(flat.mean())
    trace = []
    for step in range(steps):
        p = np.exp(logits - logits.max())
        p /= p.sum()
        arm = int(rng.choice(len(p), p=p))
        reward = float(rng.choice(rewards[arm]))
        advantage = reward - baseline
        gradient = -p.copy()
        gradient[arm] += 1
        log_ratio = np.log(p / prior)
        kl_gradient = p * (log_ratio - p @ log_ratio)
        entropy_gradient = -p * (np.log(p) - p @ np.log(p))
        logits += 0.35 * (advantage * gradient - 0.05 * kl_gradient + 0.015 * entropy_gradient)
        logits -= logits.mean()
        baseline += 0.04 * (reward - baseline)
        if step % 40 == 0 or step == steps - 1:
            trace.append(
                {
                    "step": step + 1,
                    "arm": arm,
                    "reward": reward,
                    "baseline": baseline,
                    "probabilities": p.tolist(),
                }
            )
    p = np.exp(logits - logits.max())
    p /= p.sum()
    means = np.asarray([np.mean(row) for row in rewards])
    return {
        "algorithm": "one-step REINFORCE / gradient bandit",
        "steps": steps,
        "seed": seed,
        "learning_rate": 0.35,
        "kl_coefficient": 0.05,
        "entropy_coefficient": 0.015,
        "prior": prior.tolist(),
        "probabilities": p.tolist(),
        "reward_means": means.tolist(),
        "expected_reward_before": float(prior @ means),
        "expected_reward_after": float(p @ means),
        "trace": trace,
    }


def _new_directory(output: Path):
    if output.exists():
        raise ValueError("Output already exists; use a new directory")
    output.mkdir(parents=True)


def _metrics(folder: Path) -> list[dict]:
    verify_job(folder, pixels=True)
    return json.loads((folder / "validation_report.json").read_bytes())["pages"]


def _same_math(left: list[dict], right: list[dict]):
    for a, b in zip(left, right, strict=True):
        for key in ("formula_groups", "matrix_groups", "math_fallback_glyphs", "verification"):
            if a[key] != b[key]:
                raise ValueError(f"Policy changed a protected math record: {key}")
        if a["handwriting"]["glyph_ink"] != b["handwriting"]["glyph_ink"]:
            raise ValueError("Policy changed the pen treatment")


def _ballot(root: Path, records: list[dict], seed: int):
    rng = random.Random(seed + 1009)
    edges = list(itertools.combinations(range(len(ACTIONS)), 2))
    rng.shuffle(edges)
    pairs = []
    panels = []
    font = ImageFont.load_default(size=22)
    (root / "review").mkdir()
    for index, (left, right) in enumerate(edges[:12], 1):
        if rng.random() < 0.5:
            left, right = right, left
        contexts = sorted({r["context"] for r in records})
        context = contexts[(index - 1) % len(contexts)]
        selected = [
            next(r for r in records if r["action"] == arm and r["context"] == context)
            for arm in (left, right)
        ]
        images = [Image.open(root / r["png"]).convert("RGB") for r in selected]
        boxes = [
            ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox() for im in images
        ]
        box = (
            max(0, min(b[0] for b in boxes) - 12),
            max(0, min(b[1] for b in boxes) - 12),
            max(b[2] for b in boxes) + 12,
            max(b[3] for b in boxes) + 12,
        )
        scale = min(1.4, 1100 / (box[2] - box[0]))
        size = (round((box[2] - box[0]) * scale), round((box[3] - box[1]) * scale))
        panel = Image.new("RGB", (1160, 2 * (size[1] + 55) + 45), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((15, 8), f"Pair {index:02d}   A / B / tie / neither", font=font, fill="black")
        for position, (image, label) in enumerate(zip(images, ("A", "B"), strict=True)):
            y = 50 + position * (size[1] + 55)
            draw.text((15, y), label, font=font, fill="black")
            panel.paste(image.crop(box).resize(size), (30, y + 30))
        name = f"review/{index:02d}.png"
        panel.save(root / name)
        panels.append(panel)
        pairs.append(
            {"pair_id": f"{index:02d}", "a": selected[0], "b": selected[1], "preview": name}
        )
    panels[0].save(root / "review/compare.pdf", save_all=True, append_images=panels[1:])
    atomic_json(root / "pair-map.json", pairs)
    with (root / "review/ratings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair_id", "winner", "math_error_a", "math_error_b", "notes"])
        for pair in pairs:
            writer.writerow([pair["pair_id"], "", "", "", ""])


def prepare(manifest: Path, output: Path, store: ProfileStore, *, seed: int = 73) -> dict:
    source = load_job(manifest)
    if source.render.policy is not None:
        raise ValueError("Prepare from a job without an embedded policy")
    if len(source.problems) < 3:
        raise ValueError("Supply at least three problems for separate training/evaluation content")
    _new_directory(output)
    render_job(manifest, output / "anchor", store, realism="stroke-enhanced")
    anchor = load_job(output / "anchor/manifest.json")
    initial = RenderPolicy(
        profile_sha256=anchor.profile_sha256, actions=ACTIONS, probabilities=PRIOR
    )
    atomic_json(output / "initial-policy.json", initial.model_dump(mode="json"))
    shutil.copytree(output / "anchor", output / "input")
    # Reserve the final problem(s) for a separate-content check. No human-style holdout exists yet.
    training = anchor.problems[: min(3, len(anchor.problems) - 1)]
    records = []
    seeds = (seed, seed + 64, seed + 138)
    for render_seed in seeds:
        reference = None
        for action in range(len(ACTIONS)):
            forced = initial.model_copy(
                update={"probabilities": tuple(float(k == action) for k in range(len(ACTIONS)))}
            )
            job = anchor.model_copy(deep=True)
            job.problems = training
            job.render = job.render.model_copy(update={"seed": render_seed, "policy": forced})
            tag = f"a{action}-s{render_seed}"
            path = output / f"input/{tag}.json"
            atomic_json(path, job.model_dump(mode="json"))
            folder = output / f"rollouts/{tag}"
            render_job(path, folder, store)
            pages = _metrics(folder)
            if reference is None:
                reference = pages
            else:
                _same_math(reference, pages)
            for i, page in enumerate(pages, 1):
                metrics = page["handwriting"]["policy"]
                records.append(
                    {
                        "action": action,
                        "context": f"{render_seed}:{page['id']}",
                        "png": f"rollouts/{tag}/pages/{i:02d}.png",
                        "metrics": metrics,
                        "reward": proxy_reward(metrics),
                    }
                )
    atomic_json(
        output / "dataset.json",
        {
            "seed": seed,
            "records": records,
            "renderer_sources": _renderer_stamp(),
            "training_problem_ids": [p.id or p.number for p in training],
            "evaluation_problem_ids": [p.id or p.number for p in anchor.problems[len(training) :]],
        },
    )
    _ballot(output, records, seed)
    # Editable ratings stay outside the seal; images, mappings and rollouts are sealed.
    files = {
        p.relative_to(output).as_posix(): sha256(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "ratings.csv"
    }
    atomic_json(output / "seal.json", files)
    return {
        "status": "prepared",
        "rollout_pages": len(records),
        "pairs": 12,
        "review": str(output / "review/compare.pdf"),
        "ratings": str(output / "review/ratings.csv"),
    }


def _load_trial(run: Path):
    seal = json.loads((run / "seal.json").read_bytes())
    for name, digest in seal.items():
        if sha256(inside(run, name)) != digest:
            raise ValueError(f"Trial evidence changed: {name}")
    policy = RenderPolicy.model_validate_json((run / "initial-policy.json").read_bytes())
    data = json.loads((run / "dataset.json").read_bytes())
    if data["renderer_sources"] != _renderer_stamp():
        raise ValueError("Renderer changed since this trial; prepare a new comparison")
    return policy, data


def _human_rewards(run: Path, ratings: Path) -> tuple[list[list[float]], int]:
    pairs = {p["pair_id"]: p for p in json.loads((run / "pair-map.json").read_bytes())}
    rewards = [[] for _ in ACTIONS]
    seen = set()
    valid = 0
    with ratings.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            identity = row["pair_id"].strip().zfill(2)
            if identity not in pairs or identity in seen:
                raise ValueError("Unknown or duplicate rating pair")
            seen.add(identity)
            winner = row["winner"].strip().lower()
            if not winner:
                continue
            if winner not in {"a", "b", "tie", "neither"}:
                raise ValueError("winner must be A, B, tie, or neither")
            valid += 1
            for side in ("a", "b"):
                error = row[f"math_error_{side}"].strip().lower()
                if error not in {"yes", "no"}:
                    raise ValueError("Rated pairs need math_error_a/b = yes or no")
                value = 0.5 if winner == "tie" else float(winner == side)
                if error == "yes":
                    raise ValueError(
                        f"Math error in pair {identity} {side}; investigate before training"
                    )
                rewards[pairs[identity][side]["action"]].append(value)
    if valid < 6 or not all(rewards):
        raise ValueError("Need at least six rated pairs covering all actions; twelve recommended")
    return rewards, valid


def train(
    run: Path,
    output: Path,
    store: ProfileStore,
    *,
    ratings: Path | None = None,
    seed: int = 903,
    steps: int = 800,
    parent: Path | None = None,
) -> dict:
    initial, data = _load_trial(run)
    prior = initial
    if parent:
        prior = RenderPolicy.model_validate_json(parent.read_bytes())
        if prior.profile_sha256 != initial.profile_sha256 or prior.actions != initial.actions:
            raise ValueError("Parent policy and trial actions/Profile disagree")
    rewards = [[] for _ in initial.actions]
    count = 0
    if ratings:
        rewards, count = _human_rewards(run, ratings)
    else:
        for record in data["records"]:
            rewards[record["action"]].append(record["reward"])
    fit = gradient_bandit(rewards, list(prior.probabilities), seed=seed, steps=steps)
    evidence = {
        "dataset_sha256": sha256(run / "dataset.json"),
        "seal_sha256": sha256(run / "seal.json"),
        "ratings_sha256": sha256(ratings) if ratings else None,
        "parent_sha256": sha256(parent) if parent else None,
        "human_pairs": count,
        "reward_source": "human-preference" if ratings else "proxy",
    }
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    policy = RenderPolicy(
        profile_sha256=initial.profile_sha256,
        reward_source=evidence["reward_source"],
        training_sha256=digest,
        actions=initial.actions,
        probabilities=tuple(fit["probabilities"]),
    )
    _new_directory(output)
    atomic_json(output / "policy.json", policy.model_dump(mode="json"))
    if ratings:
        shutil.copyfile(ratings, output / "ratings.csv")
    shutil.copytree(run / "anchor", output / "input")
    job = load_job(output / "input/manifest.json")
    job.render = job.render.model_copy(update={"policy": policy})
    atomic_json(output / "input/candidate.json", job.model_dump(mode="json"))
    render_job(output / "input/candidate.json", output / "candidate", store)
    _metrics(output / "candidate")
    # Check new seeds AND content excluded from the reward table.
    held = set(data["evaluation_problem_ids"])
    job.problems = [p for p in job.problems if (p.id or p.number) in held]
    evaluation = []
    for sample_seed in (seed + 1000, seed + 1001):
        paired = []
        for name, current in (("before", prior), ("after", policy)):
            job.render = job.render.model_copy(update={"seed": sample_seed, "policy": current})
            path = output / f"input/{name}-{sample_seed}.json"
            atomic_json(path, job.model_dump(mode="json"))
            folder = output / f"evaluation/{name}-{sample_seed}"
            render_job(path, folder, store)
            pages = _metrics(folder)
            paired.append(pages)
            evaluation.append(
                {
                    "phase": name,
                    "seed": sample_seed,
                    "rewards": [proxy_reward(p["handwriting"]["policy"]) for p in pages],
                }
            )
        _same_math(*paired)
    report = {
        "status": "trained",
        **evidence,
        **fit,
        "evaluation": evaluation,
        "human_naturalness": "not_evaluated",
        "scope": "renderer strategy; no new glyph generation or inferred pressure",
        "policy_sha256": sha256(output / "policy.json"),
        "candidate": str(output / "candidate/homework_solutions.pdf"),
    }
    atomic_json(output / "training.json", report)
    return report
