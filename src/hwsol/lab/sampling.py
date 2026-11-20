"""Launch pinned research checkouts in a separate, operator-provided CUDA environment."""

import os
import subprocess
import time
from pathlib import Path

from hwsol.lab.protocol import trial
from hwsol.profiles.io import atomic_json, sha256

# Paths and flags read from the pinned dataloaders and entry points.
ENTRIES = {
    "one-dm": ("test.py", "configs/IAM64.yml", "data/oov.common_words", "--one_dm"),
    "diffbrush": ("generate.py", "configs/IAM.yml", "data/wikitext103.te", "--pretrained_model"),
}


def sample(
    run: Path,
    checkout: Path,
    workdir: Path,
    torchrun: Path,
    checkpoint: Path,
    vae: Path,
    devices: int,
    dry_run: bool,
):
    settings, corpus = trial(run)
    if settings.model not in ENTRIES:
        raise ValueError("Pinned DiffusionPen single_sampling uses hardcoded words and style IDs")
    if devices < 1 or devices > len(corpus):
        raise ValueError("Device count must fit the fixed corpus")
    script, config, text_file, flag = ENTRIES[settings.model]
    raw = (run / "raw").resolve()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(checkout), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    revision = git("rev-parse", "HEAD")
    if revision != settings.revision or git("diff", "--name-only", "HEAD", "--", "*.py", config):
        raise ValueError("Sampling requires the pinned upstream code and configuration")
    for path in (
        torchrun,
        checkout / script,
        checkout / config,
        checkpoint,
        vae / "vae/config.json",
    ):
        if not path.is_file():
            raise ValueError(f"Missing local sampling prerequisite: {path}")
    actual = (workdir / text_file).read_text("utf-8").splitlines()
    if actual != list(corpus):
        raise ValueError(
            f"Place the fixed prompts.txt at {text_file} in the research work directory"
        )
    digest = sha256(checkpoint)
    if settings.checkpoint_sha256 and digest != settings.checkpoint_sha256:
        raise ValueError("Checkpoint differs from trial provenance")
    args = [
        str(torchrun.resolve()),
        "--standalone",
        f"--nproc_per_node={devices}",
        str((checkout / script).resolve()),
        "--cfg",
        str((checkout / config).resolve()),
        flag,
        str(checkpoint.resolve()),
        "--stable_dif_path",
        str(vae.resolve()),
        "--generate_type",
        "oov_u",
        "--dir",
        str(raw / "images"),
    ]
    record = {
        "status": "prepared",
        "argv": args,
        "cwd": str(workdir.resolve()),
        "revision": revision,
        "checkpoint_sha256": digest,
    }
    if dry_run:
        return record
    raw.mkdir(exist_ok=False)
    atomic_json(raw / "sampling.json", {**record, "status": "running"})
    start = time.monotonic()
    try:
        with (raw / "sampling.log").open("wb") as log:
            result = subprocess.run(
                args,
                cwd=workdir,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "WANDB_MODE": "offline",
                },
                check=False,
            )
        record["exit_code"] = result.returncode
        if result.returncode:
            raise ValueError(f"Upstream exited {result.returncode}; see raw/sampling.log")
        if not any((raw / "images").rglob("*.png")):
            raise ValueError("Upstream returned without writing any PNGs")
        record["status"] = "sampled"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        record["elapsed_seconds"] = round(time.monotonic() - start, 3)
        atomic_json(raw / "sampling.json", record)
    return record
