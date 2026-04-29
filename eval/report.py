"""
Synthetic-corpus eval report runner.
-----------------------------------
Sibling of `eval/real_doc_cases.py` (live probe) and the pytest gate at
`tests/eval/test_golden_queries.py` (binary pass/fail). This script ingests
the same synthetic corpus the pytest gate uses, runs every spec in
`tests/eval/golden_queries.yaml`, and prints aggregate metrics: overall
pass rate, latency p50/p95, per-category breakdown, and a failure listing.

The gate answers "did anything regress?" The report answers "what's the
quality picture?" — which is the artifact that makes a claim like "X% pass
rate at p95 latency Y seconds on N goldens" defensible.

Usage:
    python eval/report.py            # markdown to stdout
    python eval/report.py --json     # machine-readable JSON to stdout

Requires Ollama running locally with `nomic-embed-text` and `llama3.2`.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Make `python eval/report.py` work the same as `python -m eval.report` —
# direct invocation puts eval/ on sys.path, not the repo root, so `src` is
# unfindable without this shim. Has to run before the `src.*` imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langchain_core.documents import Document  # noqa: E402

from src.embedding import embed_documents  # noqa: E402
from src.eval import (  # noqa: E402
    format_markdown,
    load_goldens,
    run_suite,
    to_dict,
)


_DEFAULT_CORPUS = _REPO_ROOT / "tests" / "eval" / "corpus"
_DEFAULT_GOLDENS = _REPO_ROOT / "tests" / "eval" / "golden_queries.yaml"


def _ingest_corpus(corpus_dir: Path, db_dir: Path) -> None:
    """Build the same indices the pytest gate uses.

    Mirrors `tests/eval/conftest.py::ingested_db` — facts + owners on, graph
    off (the goldens target factual lookup and vector retrieval; the graph
    walk doesn't move pass/fail for any current case and adds minutes).
    Kept inline rather than imported from conftest.py because conftest is
    pytest-collection-only and the runner should work without pytest.
    """
    txt_files = sorted(corpus_dir.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No corpus files in {corpus_dir}")

    docs = [
        Document(
            page_content=path.read_text(),
            metadata={"source": str(path)},
        )
        for path in txt_files
    ]
    embed_documents(
        docs,
        persist_directory=str(db_dir),
        build_graph=False,
        build_facts=True,
        build_owners=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of markdown (for archival or downstream tools).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_DEFAULT_CORPUS,
        help="Corpus directory (default: tests/eval/corpus).",
    )
    parser.add_argument(
        "--goldens",
        type=Path,
        default=_DEFAULT_GOLDENS,
        help="Goldens YAML path (default: tests/eval/golden_queries.yaml).",
    )
    args = parser.parse_args(argv)

    specs = load_goldens(args.goldens)
    if not specs:
        print(f"No specs in {args.goldens}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="cortex_eval_") as tmp_db:
        print(
            f"Ingesting {args.corpus} → {tmp_db} ...",
            file=sys.stderr,
        )
        _ingest_corpus(args.corpus, Path(tmp_db))
        print(f"Running {len(specs)} specs ...", file=sys.stderr)
        report = run_suite(specs, tmp_db)

    if args.json:
        print(json.dumps(to_dict(report), indent=2))
    else:
        print(format_markdown(report))

    # Exit non-zero on any failure so the runner is CI-friendly even though
    # its primary purpose is informational.
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    sys.exit(main())
