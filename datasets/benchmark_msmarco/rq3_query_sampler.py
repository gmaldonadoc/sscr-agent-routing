"""Build an RQ3 query sample independently of SSR retrieval results.

This script replaces the rank-stratified sampling used by
``build_stratified_query_sample.py`` for the corrected RQ3 protocol.

Key properties
--------------
* Queries are sampled before SSR is executed.
* The sample depends only on MS MARCO queries/qrels and a fixed seed.
* No query is filtered according to SSR rank, Recall@K, or retrieval success.
* The output records enough metadata to reproduce the sample exactly.

The original sampling script is intentionally left untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES_PATH = BASE_DIR / "queries.dev.small.tsv"
DEFAULT_QRELS_PATH = BASE_DIR / "qrels.dev.small.tsv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "rq3_query_sample.json"
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 42


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries[str(row[0])] = row[1].strip()
    return queries


def load_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            try:
                relevance = int(row[3])
            except ValueError:
                continue
            if relevance > 0:
                qrels[str(row[0])].append(str(row[2]))
    return {qid: sorted(set(pids)) for qid, pids in qrels.items()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_queries(
    queries: dict[str, str],
    qrels: dict[str, list[str]],
    sample_size: int,
    seed: int,
) -> list[dict]:
    eligible_qids = sorted(
        qid
        for qid in queries
        if qid in qrels and len(qrels[qid]) > 0
    )

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    if sample_size > len(eligible_qids):
        raise ValueError(
            f"Requested {sample_size} queries, but only "
            f"{len(eligible_qids)} queries have at least one positive qrel."
        )

    rng = random.Random(seed)
    selected_qids = rng.sample(eligible_qids, sample_size)

    # Preserve the deterministic randomized order produced by rng.sample.
    return [
        {
            "query_id": qid,
            "query": queries[qid],
            "positive_ids": qrels[qid],
        }
        for qid in selected_qids
    ]


def build_output(
    selected: Iterable[dict],
    queries_path: Path,
    qrels_path: Path,
    sample_size: int,
    seed: int,
    eligible_count: int,
) -> dict:
    selected_list = list(selected)
    return {
        "metadata": {
            "selection_method": "uniform_random_independent_of_ssr",
            "selection_depends_on_ssr_output": False,
            "seed": seed,
            "requested_sample_size": sample_size,
            "selected_total": len(selected_list),
            "eligible_queries": eligible_count,
            "eligibility_rule": (
                "query must exist in the MS MARCO query file and have at "
                "least one qrel with relevance > 0"
            ),
            "queries_path": str(queries_path),
            "qrels_path": str(qrels_path),
            "queries_sha256": file_sha256(queries_path),
            "qrels_sha256": file_sha256(qrels_path),
        },
        "queries": selected_list,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible MS MARCO RQ3 query sample without using "
            "SSR ranks or retrieval results."
        )
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"MS MARCO query TSV (default: {DEFAULT_QUERIES_PATH})",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=DEFAULT_QRELS_PATH,
        help=f"MS MARCO qrels TSV (default: {DEFAULT_QRELS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of queries to sample (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Sampling seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    eligible_count = sum(
        1 for qid in queries if qid in qrels and len(qrels[qid]) > 0
    )

    selected = sample_queries(
        queries=queries,
        qrels=qrels,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    output = build_output(
        selected=selected,
        queries_path=args.queries,
        qrels_path=args.qrels,
        sample_size=args.sample_size,
        seed=args.seed,
        eligible_count=eligible_count,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {len(selected)} independently sampled queries to:")
    print(args.output)
    print(json.dumps(output["metadata"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
