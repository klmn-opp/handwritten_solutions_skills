"""Optional text-only line backends. Production mathematics never calls this module."""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from hwsol.profiles.io import atomic_json, sha256


@dataclass(frozen=True)
class TextLineRequest:
    text: str
    reference: Path
    output: Path
    seed: int = 42

    def validate(self) -> None:
        if not self.text.strip() or any(ch in self.text for ch in "$\\\n="):
            raise ValueError("TextLineRenderer accepts one prose line, never mathematics")
        if not self.reference.is_file():
            raise ValueError("Supply a real personal reference line")
        if self.output.exists():
            raise FileExistsError(self.output)


class TextLineRenderer(Protocol):
    def render(self, request: TextLineRequest) -> dict: ...


class RecordedWordRenderer:
    """CPU composition of complete observed words; never invent a connecting stroke."""

    def __init__(self, traces: tuple, source_sha256: str):
        self.words = {t.text: t for t in traces if t.kind in {"word", "line"}}
        self.source_sha256 = source_sha256

    def render(self, request: TextLineRequest) -> dict:
        request.validate()
        words = request.text.split()
        missing = [word for word in words if word not in self.words]
        if missing:
            return {
                "status": "unavailable",
                "backend": "recorded-word-cpu",
                "missing_words": missing,
                "fallback": "personal-font-runtime",
            }
        parts = []
        width = 8
        # Samples share one source coordinate space; do not independently enlarge short
        # words or erase the relative scale of words containing descenders.
        selected_points = [
            p for word in words for stroke in self.words[word].strokes for p in stroke
        ]
        common_height = max(p.y for p in selected_points) - min(p.y for p in selected_points)
        scale = 48 / max(1, common_height)
        common_top = min(p.y for p in selected_points)
        for word in words:
            trace = self.words[word]
            points = [p for stroke in trace.strokes for p in stroke]
            x0, x1 = min(p.x for p in points), max(p.x for p in points)
            parts.append((trace, x0, common_top, scale, width))
            width += round((x1 - x0) * scale) + 16
        image = Image.new("L", (width, 64), 255)
        draw = ImageDraw.Draw(image)
        for trace, x0, y0, scale, offset in parts:
            for stroke in trace.strokes:
                xy = [(offset + (p.x - x0) * scale, 8 + (p.y - y0) * scale) for p in stroke]
                draw.line(xy, fill=0, width=2, joint="curve")
        request.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(request.output)
        return {
            "status": "composed-unvalidated",
            "backend": "recorded-word-cpu",
            "source_sha256": self.source_sha256,
            "origins": [self.words[word].origin for word in words],
            "output_sha256": sha256(request.output),
            "content_validation": "not-evaluated",
        }


class HandnoteLineRenderer:
    """Run the real locked DiffBrush CLI in a separately supplied Python environment."""

    def __init__(self, python: Path, *, device: str = "cpu", steps: int = 50, timeout: int = 600):
        self.python, self.device, self.steps, self.timeout = python, device, steps, timeout

    def render(self, request: TextLineRequest) -> dict:
        request.validate()
        request.output.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self.python),
            "-m",
            "handnote",
            "line",
            "generate",
            "--text",
            request.text,
            "--style",
            str(request.reference.resolve()),
            "--out",
            str(request.output.resolve()),
            "--device",
            self.device,
            "--steps",
            str(self.steps),
            "--seed",
            str(request.seed),
        ]
        report = {
            "backend": "handnote-diffbrush",
            "mock": False,
            "argv": argv,
            "reference_sha256": sha256(request.reference),
            "text": request.text,
            "seed": request.seed,
            "status": "failed",
            "content_validation": "not-evaluated",
            "style_naturalness": None,
            "cer": None,
            "wer": None,
            "inference_seconds": None,
            "peak_memory_bytes": None,
        }
        start = time.monotonic()
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=self.timeout, check=False
            )
            report.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if result.returncode == 0 and request.output.is_file():
                with Image.open(request.output) as image:
                    image.verify()
                report.update(status="generated-unvalidated", output_sha256=sha256(request.output))
        except subprocess.TimeoutExpired:
            report["error"] = "inference-timeout"
        finally:
            report["process_seconds"] = time.monotonic() - start
            atomic_json(request.output.with_suffix(".experiment.json"), report)
        return report


def transcription_scores(expected: str, actual: str) -> dict:
    """Independent OCR/human transcription; never use the requested prompt as OCR output."""

    def edits(a, b):
        row = list(range(len(b) + 1))
        for i, left in enumerate(a, 1):
            following = [i]
            for j, right in enumerate(b, 1):
                following.append(
                    min(row[j] + 1, following[j - 1] + 1, row[j - 1] + (left != right))
                )
            row = following
        return row[-1]

    if not expected:
        raise ValueError("Empty expected transcription")
    words = expected.split()
    return {
        "cer": edits(expected, actual) / len(expected),
        "wer": edits(words, actual.split()) / max(1, len(words)),
        "exact_match": expected == actual,
        "length_delta": len(actual) - len(expected),
        "word_count_delta": len(actual.split()) - len(words),
    }
