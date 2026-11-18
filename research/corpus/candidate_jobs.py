"""Bounded, local formula-recognition jobs; never modify review decisions."""

import threading
import uuid
from pathlib import Path

from research.corpus.recognition import recognize
from research.corpus.store import Corpus


def select_missing(corpus: Corpus, page_id: str) -> list[str]:
    corpus.get("page", page_id)
    recognized = {
        r["sample_id"]
        for r in corpus.rows("recognition")
        if r["engine"] == "unimernet" and r["text"].strip()
    }
    return [
        s["sample_id"]
        for s in corpus.rows("sample")
        if s["page_id"] == page_id
        and s["review_status"] not in {"verified", "rejected"}
        and s["content_type"] in {"unknown", "math-expression", "math-symbol"}
        and s["sample_id"] not in recognized
    ]


class CandidateJobs:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        root = Path(__file__).resolve().parents[2]
        self.python = root / "workspace/environments/corpus-neural-v09/bin/python"
        self.checkpoint = root / "workspace/checkpoints/unimernet-tiny-v09"
        self.source = root / "workspace/sources/v09/UniMERNet"
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def available(self) -> bool:
        return (
            self.python.is_file()
            and (self.checkpoint / "unimernet_tiny.pth").is_file()
            and (self.source / "unimernet/models/unimernet/unimernet.py").is_file()
        )

    def start(self, *, sample_id: str | None = None, page_id: str | None = None) -> dict:
        if bool(sample_id) == bool(page_id):
            raise ValueError("Choose one region or one page")
        if not self.available():
            raise ValueError("本地公式识别环境或权重未安装；已保存候选仍可使用")
        if sample_id:
            sample = self.corpus.get("sample", sample_id)
            if sample["review_status"] == "rejected":
                raise ValueError("已排除区域不自动识别；请先选择需要审核的区域")
            ids = [sample_id]
        else:
            ids = select_missing(self.corpus, page_id)
        if not ids:
            return {"status": "complete", "count": 0, "message": "本页待审核区域已有公式候选"}
        if len(ids) > 100:
            raise ValueError("一次最多识别100个区域，请先逐区域识别")
        with self.lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise ValueError("已有本地识别任务运行，请等待完成后重试")
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "status": "running", "count": len(ids)}
            self.jobs[job_id] = job
            for key in list(self.jobs)[:-20]:
                del self.jobs[key]
            threading.Thread(target=self._work, args=(job_id, ids), daemon=True).start()
            return dict(job)

    def _work(self, job_id: str, ids: list[str]):
        try:
            run = recognize(
                self.corpus,
                "unimernet",
                ids,
                python=self.python,
                checkpoint=self.checkpoint,
                source=self.source,
                candidate_count=3,
            )
            update = {"status": "complete", "run": str(run)}
        except Exception as exc:
            update = {"status": "failed", "error": str(exc)}
        with self.lock:
            self.jobs[job_id].update(update)

    def get(self, job_id: str) -> dict:
        with self.lock:
            return dict(self.jobs[job_id])
