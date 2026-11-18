"""Research CLI; all mutating experiments receive canonical run receipts."""

import argparse
import json
from pathlib import Path

from hwsol.profiles.io import atomic_json
from research.corpus.experiment import create
from research.corpus.store import Corpus


def main():
    parser = argparse.ArgumentParser(description="Private historical homework corpus")
    parser.add_argument("--corpus", type=Path, default=Path("workspace/datasets/personal-homework"))
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("source", type=Path, nargs="?", default=Path("source_pdf"))
    imp.add_argument(
        "--batch", required=True, help="Real collection session, or document-specific batch"
    )
    imp.add_argument("--course", default="unspecified")
    imp.add_argument("--pages", help="1-based comma-separated pages; omitted imports all pages")
    imp.add_argument("--dpi", type=int, default=150)
    imp.add_argument("--run-id")
    imp.add_argument("--propose-lines", action="store_true")
    studio = sub.add_parser("review")
    studio.add_argument("--port", type=int, default=8769)
    studio.add_argument("--katex", type=Path, default=Path("workspace/sources/v09/katex/dist"))
    export = sub.add_parser("export")
    export.add_argument("--run-id")
    recognize = sub.add_parser("recognize")
    recognize.add_argument("--engine", choices=["tesseract", "trocr", "unimernet"], required=True)
    recognize.add_argument(
        "--ids", type=Path, required=True, help="JSON array of explicit sample IDs"
    )
    recognize.add_argument("--python", type=Path)
    recognize.add_argument("--checkpoint", type=Path)
    recognize.add_argument("--source", type=Path)
    recognize.add_argument("--run-id")
    sub.add_parser("status")
    args = parser.parse_args()
    corpus = Corpus(args.corpus)
    if args.command == "review":
        from research.corpus.studio import server

        service = server(corpus, port=args.port, katex=args.katex)
        print(f"http://127.0.0.1:{service.server_port} (local only)", flush=True)
        try:
            service.serve_forever()
        finally:
            service.server_close()
        return
    if args.command == "recognize":
        from research.corpus.recognition import recognize

        print(
            recognize(
                corpus,
                args.engine,
                json.loads(args.ids.read_text()),
                run_id=args.run_id,
                python=args.python,
                checkpoint=args.checkpoint,
                source=args.source,
            )
        )
        return
    if args.command == "status":
        from collections import Counter

        print(
            json.dumps(
                {
                    "documents": len(corpus.rows("document")),
                    "rendered_pages": len(corpus.rows("page")),
                    "sample_status": dict(
                        Counter(s["review_status"] for s in corpus.rows("sample"))
                    ),
                },
                indent=2,
            )
        )
        return
    run = create(
        "corpus-" + args.command,
        run_id=args.run_id,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    try:
        if args.command == "import":
            from research.corpus.ingest import EXTENSIONS, import_file, propose_lines

            paths = (
                [args.source]
                if args.source.is_file()
                else sorted(p for p in args.source.rglob("*") if p.suffix.lower() in EXTENSIONS)
            )
            if not paths:
                raise ValueError("No PDF/image input files found")
            indices = [int(i) - 1 for i in args.pages.split(",")] if args.pages else None
            result = []
            for path in paths:
                doc = import_file(
                    corpus,
                    path,
                    course=args.course,
                    batch=args.batch,
                    dpi=args.dpi,
                    page_indices=indices,
                )
                result.append(doc)
                if args.propose_lines:
                    for page in corpus.rows("page"):
                        if page["document_id"] == doc["document_id"]:
                            propose_lines(corpus, page["page_id"])
        else:
            from research.corpus.dataset import export_dataset

            result = export_dataset(corpus, run.path / "reports/dataset")
        atomic_json(run.path / "reports/result.json", result)
        run.finish(
            "passed" if args.command == "import" or result["eligible_samples"] else "held",
            result={"operation": args.command, "production_promotion": False},
        )
        print(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
