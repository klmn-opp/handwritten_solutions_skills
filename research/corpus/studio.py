"""Loopback-only annotation studio. No external services, scripts or automatic verification."""

import json
import mimetypes
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from research.corpus.dataset import export_dataset
from research.corpus.experiment import create
from research.corpus.store import Conflict, Corpus


def server(corpus: Corpus, *, port: int = 8769, katex: Path | None = None) -> ThreadingHTTPServer:
    token = secrets.token_urlsafe(32)
    static = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data: bytes, content_type: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def json(self, obj, status=200):
            self.reply(
                json.dumps(obj, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
                status,
            )

        def local(self):
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def do_GET(self):
            if not self.local():
                return self.json({"error": "Loopback host required"}, 403)
            url = urlparse(self.path)
            try:
                if url.path == "/api/state":
                    return self.json(
                        {
                            "documents": corpus.rows("document"),
                            "pages": corpus.rows("page"),
                            "samples": corpus.rows("sample"),
                            "recognition": corpus.rows("recognition"),
                        }
                    )
                if url.path == "/api/history":
                    return self.json(corpus.history(parse_qs(url.query)["id"][0]))
                if url.path == "/asset":
                    file = corpus.path(parse_qs(url.query)["path"][0])
                elif url.path.startswith("/katex/"):
                    if katex is None:
                        raise FileNotFoundError("Install local KaTeX for offline math preview")
                    file = (katex / url.path.removeprefix("/katex/")).resolve()
                    if not file.is_relative_to(katex.resolve()):
                        raise ValueError("Invalid static path")
                else:
                    name = "index.html" if url.path == "/" else url.path.removeprefix("/")
                    if name not in {"index.html", "studio.js", "studio.css"}:
                        raise FileNotFoundError(name)
                    file = static / name
                data = file.read_bytes()
                if file.name == "index.html":
                    data = data.replace(b"__CSRF__", token.encode())
                self.reply(data, mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self.json({"error": str(exc)}, 404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (
                not self.local()
                or self.headers.get("X-Corpus-Token") != token
                or (origin and origin != f"http://{self.headers.get('Host')}")
            ):
                return self.json({"error": "Local authenticated review required"}, 403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if (
                    not 0 < size <= 1_000_000
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ValueError("Expected bounded JSON request")
                req = json.loads(self.rfile.read(size))
                if self.path == "/api/review":
                    result = corpus.review(
                        req["sample_id"],
                        req["revision"],
                        req["changes"],
                        req["actor"],
                        req.get("attestation"),
                    )
                elif self.path == "/api/reshape":
                    result = corpus.reshape(req["items"], req.get("boxes"), req["actor"])
                elif self.path == "/api/propose":
                    if not req.get("actor", "").strip():
                        raise ValueError("Reviewer required")
                    result = corpus.propose(
                        req["page_id"], req["box"], label_source="manual-region:" + req["actor"]
                    )
                elif self.path == "/api/export":
                    run = create(
                        "corpus-reviewed-export",
                        config={"corpus": str(corpus.root), "actor": req["actor"]},
                    )
                    try:
                        result = export_dataset(corpus, run.path / "reports/dataset")
                        run.finish(
                            "passed" if result["eligible_samples"] else "held", result=result
                        )
                        result["path"] = str(run.path)
                    except Exception as exc:
                        run.fail(exc)
                        raise
                else:
                    return self.json({"error": "Unknown action"}, 404)
                self.json(result)
            except Conflict as exc:
                self.json({"error": str(exc)}, 409)
            except (ValueError, KeyError, TypeError) as exc:
                self.json({"error": str(exc)}, 400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
