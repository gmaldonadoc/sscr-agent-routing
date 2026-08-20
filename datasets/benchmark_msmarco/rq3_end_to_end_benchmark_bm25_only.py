"""Corrected RQ3 end-to-end benchmark for SSR on MS MARCO.

This script is intentionally separate from ``end_to_end_orchestrator_benchmark.py``
so that the original experiment remains unchanged.

Corrections implemented here
----------------------------
1. Uses a query sample created independently of SSR ranks/results.
2. Builds nested registries: A_100 subset A_500 subset A_1000 subset A_5000.
3. Uses an explicit fixed top-K retrieval budget for SSR.
4. Never falls back to the full catalog when SSR returns no candidates.
5. Separates offline index-construction time from online retrieval time.
6. Reports candidate recall, conditional LLM selection accuracy, and end-to-end
   accuracy separately.
7. Uses one configurable LLM/pricing configuration for all methods and records
   it in experiment metadata.
8. Uses API token accounting when available, with a clearly marked character
   estimate fallback.
9. Reports successful inference latency separately from time-to-failure.
10. Records mean/median/p95 latency and other reproducibility metadata.
11. Uses 1-based candidate-index selection from the LLM, maps the index to the
    passage ID internally, validates the selected index against pool size, and
    stores the raw LLM response for auditability.
12. Adds a bounded BM25 baseline using the same normalized unigram representation
    and SQLite FTS5 bm25() ranking used in RQ2.
13. Uses the same maximum candidate budget for SSR and BM25 and never pads either
    retriever with non-matching passages.

Typical usage
-------------
First create an independent sample:

    python datasets/benchmark_msmarco/rq3_query_sampler.py \
        --sample-size 200 --seed 42

Then run a retrieval-only smoke test (no API calls):

    python datasets/benchmark_msmarco/rq3_end_to_end_benchmark_bm25.py \
        --skip-llm --skip-traditional --max-queries 5 \
        --n-values 100,500 --k-values 5,10,20,50

Run a small end-to-end smoke test with the paper budget K=20:

    python datasets/benchmark_msmarco/rq3_end_to_end_benchmark_bm25.py \
        --skip-traditional --max-queries 2 --n-values 100,500 --k-values 20

For the final paper run, use a single fixed K selected by the validation
protocol and evaluate Traditional, SSR, and BM25 under the same run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from utilities.routing.msmarco_indexer import MSMarcoIndexer  # noqa: E402


DEFAULT_COLLECTION_PATH = BASE_DIR / "collection.tsv"
DEFAULT_QUERY_SAMPLE_PATH = BASE_DIR / "rq3_query_sample.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "rq3_bm25"
DEFAULT_CACHE_PATH = BASE_DIR / "rq3_docs_cache.pkl"
DEFAULT_N_VALUES = [100, 500, 1000, 5000]
DEFAULT_K_VALUES = [5, 10, 20, 50]
DEFAULT_SEED = 42
DEFAULT_MAX_NEGATIVE_DOCS = 50_000
DEFAULT_MODEL = os.getenv("RQ3_MODEL", "gpt-4o-mini")
DEFAULT_INPUT_PRICE_PER_1M = float(os.getenv("RQ3_INPUT_PRICE_PER_1M", "0.15"))
DEFAULT_OUTPUT_PRICE_PER_1M = float(os.getenv("RQ3_OUTPUT_PRICE_PER_1M", "0.60"))


@dataclass
class BM25RegistryIndex:
    connection: sqlite3.Connection
    text_by_pid: dict[str, str]


@dataclass
class SSRRegistryIndex:
    features: dict[str, Any]
    signal_idf: dict[str, float]
    signal_index: dict[str, set[str]]
    text_by_pid: dict[str, str]


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(v) for v in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def estimate_tokens(text: str) -> int:
    # Fallback only. Successful API responses should use provider usage fields.
    return max(1, len(text) // 4)


def compute_estimated_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    return (
        input_tokens / 1_000_000 * input_price_per_1m
        + output_tokens / 1_000_000 * output_price_per_1m
    )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean_or_none(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def median_or_none(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else None


def stdev_or_zero(values: Iterable[float | int | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return statistics.stdev(clean) if len(clean) > 1 else 0.0


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def load_query_sample(path: Path) -> tuple[dict, list[dict]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metadata = data.get("metadata", {})
    queries = data.get("queries", [])

    if not queries:
        raise ValueError(f"No queries found in {path}")

    depends_on_ssr = metadata.get("selection_depends_on_ssr_output")
    method = str(metadata.get("selection_method", "")).lower()

    if depends_on_ssr is not False and "independent_of_ssr" not in method:
        raise ValueError(
            "RQ3 requires a query sample selected independently of SSR output. "
            "Generate one with rq3_query_sampler.py instead of using the old "
            "rank-stratified sample."
        )

    normalized = []
    for item in queries:
        qid = str(item["query_id"])
        positive_ids = sorted({str(pid) for pid in item.get("positive_ids", [])})
        if not positive_ids:
            raise ValueError(f"Query {qid} has no positive_ids")
        normalized.append(
            {
                "query_id": qid,
                "query": str(item["query"]).strip(),
                "positive_ids": positive_ids,
            }
        )

    return metadata, normalized


def _cache_is_valid(
    cache_payload: dict,
    collection_path: Path,
    required_pids: set[str],
    max_negative_docs: int,
    negative_seed: int,
) -> bool:
    metadata = cache_payload.get("metadata", {})
    docs = cache_payload.get("docs")
    if not isinstance(docs, dict):
        return False

    try:
        collection_size = collection_path.stat().st_size
    except FileNotFoundError:
        return False

    return (
        metadata.get("collection_size") == collection_size
        and metadata.get("required_pids_hash") == stable_id_hash(required_pids)
        and metadata.get("max_negative_docs") == max_negative_docs
        and metadata.get("negative_seed") == negative_seed
        and required_pids.issubset(docs.keys())
    )


def load_documents_with_reservoir_negatives(
    collection_path: Path,
    required_pids: set[str],
    max_negative_docs: int,
    negative_seed: int,
    cache_path: Path | None,
) -> dict[str, str]:
    """Load all required positives plus a reproducible reservoir of negatives."""

    if not collection_path.exists():
        raise FileNotFoundError(
            f"MS MARCO collection not found: {collection_path}. "
            "Place collection.tsv there or pass --collection."
        )

    if cache_path and cache_path.exists():
        with cache_path.open("rb") as handle:
            cache_payload = pickle.load(handle)
        if _cache_is_valid(
            cache_payload,
            collection_path,
            required_pids,
            max_negative_docs,
            negative_seed,
        ):
            print(f"Loading validated document cache: {cache_path}")
            return cache_payload["docs"]
        print("Existing document cache does not match this experiment; rebuilding it.")

    rng = random.Random(negative_seed)
    positive_docs: dict[str, str] = {}
    reservoir: list[tuple[str, str]] = []
    negative_seen = 0

    print("Scanning collection and building reproducible negative reservoir...")
    with collection_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_index, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue

            pid = str(row[0])
            text = row[1].strip()

            if pid in required_pids:
                positive_docs[pid] = text
                continue

            negative_seen += 1
            if len(reservoir) < max_negative_docs:
                reservoir.append((pid, text))
            else:
                replacement = rng.randrange(negative_seen)
                if replacement < max_negative_docs:
                    reservoir[replacement] = (pid, text)

            if row_index % 1_000_000 == 0:
                print(
                    f"  scanned={row_index:,}, positives={len(positive_docs):,}, "
                    f"negative_reservoir={len(reservoir):,}"
                )

    missing = sorted(required_pids.difference(positive_docs))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{len(missing)} required positive documents were not found in the "
            f"collection. First missing IDs: {preview}"
        )

    docs = dict(reservoir)
    docs.update(positive_docs)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "collection_path": str(collection_path),
                "collection_size": collection_path.stat().st_size,
                "required_pids_hash": stable_id_hash(required_pids),
                "max_negative_docs": max_negative_docs,
                "negative_seed": negative_seed,
                "negative_sampling": "reservoir_sampling",
                "required_positive_count": len(required_pids),
            },
            "docs": docs,
        }
        with cache_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved document cache: {cache_path}")

    return docs


def build_nested_candidate_pools(
    qid: str,
    positive_ids: list[str],
    docs: dict[str, str],
    n_values: list[int],
    seed: int,
) -> dict[int, list[dict]]:
    positives = [pid for pid in positive_ids if pid in docs]
    if not positives:
        raise ValueError(f"No positive document for query {qid} is available in docs")

    max_n = max(n_values)
    if any(n < len(positives) for n in n_values):
        raise ValueError(
            f"At least one N is smaller than the number of positives for query {qid}"
        )

    positive_set = set(positives)
    negative_ids = [pid for pid in docs if pid not in positive_set]

    rng = random.Random(f"{seed}:{qid}:negative-order")
    rng.shuffle(negative_ids)

    negatives_needed = max_n - len(positives)
    if len(negative_ids) < negatives_needed:
        raise ValueError(
            f"Not enough negative documents for query {qid}: need "
            f"{negatives_needed}, have {len(negative_ids)}"
        )

    pools: dict[int, list[dict]] = {}
    for n in n_values:
        selected_ids = positives + negative_ids[: n - len(positives)]

        # Randomize presentation order for the full-catalog baseline without
        # changing the nested membership of the registries.
        display_rng = random.Random(f"{seed}:{qid}:display:{n}")
        display_ids = list(selected_ids)
        display_rng.shuffle(display_ids)

        pools[n] = [{"pid": pid, "text": docs[pid]} for pid in display_ids]

    # Defensive check: membership must be nested even though presentation order
    # differs at each N.
    for smaller, larger in zip(n_values, n_values[1:]):
        small_ids = {doc["pid"] for doc in pools[smaller]}
        large_ids = {doc["pid"] for doc in pools[larger]}
        if not small_ids.issubset(large_ids):
            raise AssertionError(f"Registry nesting failed for query {qid}")

    return pools


def ensure_fts5(connection: sqlite3.Connection) -> None:
    """Fail clearly if the local SQLite build does not provide FTS5."""
    try:
        connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_test USING fts5(x)")
        connection.execute("DROP TABLE IF EXISTS __fts5_test")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "This Python/SQLite build does not provide FTS5, required for the BM25 baseline."
        ) from exc


def fts_quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def build_bm25_registry_index(
    candidates: list[dict],
    indexer: MSMarcoIndexer,
) -> tuple[BM25RegistryIndex, float]:
    """Offline BM25 phase using the same normalized-unigram + FTS5 design as RQ2."""
    start = time.perf_counter()
    connection = sqlite3.connect(":memory:")
    ensure_fts5(connection)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        "CREATE VIRTUAL TABLE docs USING fts5("
        "pid UNINDEXED, terms, tokenize='unicode61 remove_diacritics 0')"
    )

    text_by_pid: dict[str, str] = {}
    rows: list[tuple[str, str]] = []
    for doc in candidates:
        pid = str(doc["pid"])
        text = str(doc["text"])
        # Match the RQ2 BM25 baseline: normalized unigrams only.
        terms = " ".join(indexer._tokenize(text))
        rows.append((pid, terms))
        text_by_pid[pid] = text

    connection.executemany("INSERT INTO docs(pid, terms) VALUES (?, ?)", rows)
    connection.commit()
    connection.execute("INSERT INTO docs(docs) VALUES ('optimize')")
    connection.commit()

    return (
        BM25RegistryIndex(connection=connection, text_by_pid=text_by_pid),
        time.perf_counter() - start,
    )


def retrieve_bm25_ranked(
    query: str,
    registry_index: BM25RegistryIndex,
    indexer: MSMarcoIndexer,
    top_k: int,
) -> tuple[list[dict], float, int]:
    """Online BM25 retrieval with no padding of non-matching passages."""
    start = time.perf_counter()
    tokens = indexer._tokenize(query)
    unique_tokens = list(dict.fromkeys(tokens))
    if not unique_tokens:
        return [], time.perf_counter() - start, 0

    match_expr = " OR ".join(fts_quote(token) for token in unique_tokens)
    rows = registry_index.connection.execute(
        "SELECT pid, bm25(docs) AS score FROM docs "
        "WHERE docs MATCH ? ORDER BY score ASC, pid ASC LIMIT ?",
        (match_expr, int(top_k)),
    ).fetchall()

    ranked = [
        {
            "pid": str(pid),
            "text": registry_index.text_by_pid[str(pid)],
            # SQLite FTS5 returns lower (more negative) scores for better matches.
            "bm25_score": float(score),
        }
        for pid, score in rows
    ]
    return ranked, time.perf_counter() - start, len(unique_tokens)


def compute_signal_idf(features: dict[str, Any]) -> dict[str, float]:
    signal_df: dict[str, int] = defaultdict(int)
    total_docs = len(features)
    for feature in features.values():
        for signal in feature.graph_signals:
            signal_df[signal] += 1

    return {
        signal: math.log((total_docs + 1) / (df + 1)) + 1.0
        for signal, df in signal_df.items()
    }


def build_signal_index(features: dict[str, Any]) -> dict[str, set[str]]:
    signal_index: dict[str, set[str]] = defaultdict(set)
    for pid, feature in features.items():
        for signal in feature.graph_signals:
            signal_index[signal].add(pid)
    return dict(signal_index)


def build_ssr_registry_index(
    candidates: list[dict],
    indexer: MSMarcoIndexer,
) -> tuple[SSRRegistryIndex, float]:
    """Offline phase: feature extraction, IDF, and inverted-index creation."""
    start = time.perf_counter()

    features: dict[str, Any] = {}
    text_by_pid: dict[str, str] = {}
    for doc in candidates:
        feature = indexer.index_document({"id": doc["pid"], "text": doc["text"]})
        features[feature.agent_name] = feature
        text_by_pid[doc["pid"]] = doc["text"]

    registry_index = SSRRegistryIndex(
        features=features,
        signal_idf=compute_signal_idf(features),
        signal_index=build_signal_index(features),
        text_by_pid=text_by_pid,
    )
    return registry_index, time.perf_counter() - start


def retrieve_ssr_ranked(
    query: str,
    registry_index: SSRRegistryIndex,
    indexer: MSMarcoIndexer,
) -> tuple[list[dict], float, int]:
    """Online phase: query extraction, candidate lookup, scoring, and ranking."""
    start = time.perf_counter()

    query_signals = indexer.extract_query_signals(query)
    candidate_ids: set[str] = set()
    for signal in query_signals:
        candidate_ids.update(registry_index.signal_index.get(signal, set()))

    ranked: list[dict] = []
    for pid in candidate_ids:
        feature = registry_index.features[pid]
        matched = query_signals.intersection(feature.graph_signals)
        if not matched:
            continue

        score = sum(registry_index.signal_idf.get(signal, 1.0) for signal in matched)
        ranked.append(
            {
                "pid": pid,
                "text": registry_index.text_by_pid[pid],
                "ssr_score": score,
                "matched_signals": sorted(matched),
            }
        )

    ranked.sort(
        key=lambda item: (
            -item["ssr_score"],
            -len(item["matched_signals"]),
            item["pid"],
        )
    )

    latency = time.perf_counter() - start
    return ranked, latency, len(query_signals)


def build_prompt(query: str, candidates: list[dict]) -> str:
    docs_text = []
    for idx, doc in enumerate(candidates, start=1):
        docs_text.append(
            f"Candidate {idx}\n"
            f"text: {doc['text']}\n"
        )

    return (
        "You are a retrieval orchestrator.\n"
        "Given a user query and candidate passages, select the single passage "
        "that best answers the query.\n\n"
        f"The selected_index must be an integer between 1 and {len(candidates)}.\n"
        "Return only valid JSON in this format:\n"
        '{"selected_index": 1}\n\n'
        f"Query:\n{query}\n\n"
        "Candidate passages:\n"
        + "\n".join(docs_text)
    )


def _coerce_selected_index(value: Any) -> int | None:
    """Parse a model-produced candidate index without silently truncating it."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("+"):
            stripped = stripped[1:]
        if stripped.isdigit():
            return int(stripped)
    return None


def _usage_token_value(usage: Any, preferred: str, fallback: str) -> int | None:
    if usage is None:
        return None
    value = getattr(usage, preferred, None)
    if value is None:
        value = getattr(usage, fallback, None)
    return int(value) if value is not None else None


def call_orchestrator(
    client: Any,
    model: str,
    prompt: str,
    candidates: list[dict],
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> dict:
    start = time.perf_counter()
    estimated_input_tokens = estimate_tokens(prompt)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        attempt_latency = time.perf_counter() - start
        content = response.choices[0].message.content or ""

        usage = getattr(response, "usage", None)
        input_tokens = _usage_token_value(usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_token_value(usage, "completion_tokens", "output_tokens")
        token_source = "api_usage"

        if input_tokens is None:
            input_tokens = estimated_input_tokens
            token_source = "char_estimate"
        if output_tokens is None:
            output_tokens = estimate_tokens(content) if content else 0
            token_source = "char_estimate"

        selected_index = None
        selected_pid = None
        selection_parse_error = None
        try:
            parsed = json.loads(content)
            selected_index = _coerce_selected_index(parsed.get("selected_index"))
            if selected_index is None:
                selection_parse_error = "missing_or_non_integer_selected_index"
            elif not 1 <= selected_index <= len(candidates):
                selection_parse_error = "selected_index_out_of_range"
            else:
                selected_pid = str(candidates[selected_index - 1]["pid"])
        except Exception as exc:
            selection_parse_error = f"invalid_json: {exc}"

        return {
            "llm_attempted": 1,
            "api_success": 1,
            "api_error": None,
            "raw_llm_response": content,
            "selected_index": selected_index,
            "selection_parse_error": selection_parse_error,
            "selected_pid": selected_pid,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "token_count_source": token_source,
            "estimated_cost_usd": compute_estimated_cost(
                input_tokens,
                output_tokens,
                input_price_per_1m,
                output_price_per_1m,
            ),
            "llm_latency_s": attempt_latency,
            "failure_latency_s": None,
        }

    except Exception as exc:
        failure_latency = time.perf_counter() - start
        return {
            "llm_attempted": 1,
            "api_success": 0,
            "api_error": str(exc),
            "raw_llm_response": None,
            "selected_index": None,
            "selection_parse_error": None,
            "selected_pid": None,
            "input_tokens": estimated_input_tokens,
            "output_tokens": 0,
            "token_count_source": "char_estimate_after_api_failure",
            "estimated_cost_usd": compute_estimated_cost(
                estimated_input_tokens,
                0,
                input_price_per_1m,
                output_price_per_1m,
            ),
            "llm_latency_s": None,
            "failure_latency_s": failure_latency,
        }


def skipped_llm_result(prompt: str | None) -> dict:
    input_tokens = estimate_tokens(prompt) if prompt else 0
    return {
        "llm_attempted": 0,
        "api_success": None,
        "api_error": None,
        "raw_llm_response": None,
        "selected_index": None,
        "selection_parse_error": None,
        "selected_pid": None,
        "input_tokens": input_tokens,
        "output_tokens": 0,
        "token_count_source": "char_estimate_skip_llm",
        "estimated_cost_usd": None,
        "llm_latency_s": None,
        "failure_latency_s": None,
    }


def evaluate_llm_pool(
    query: str,
    candidates: list[dict],
    positive_ids: set[str],
    client: Any | None,
    model: str,
    input_price_per_1m: float,
    output_price_per_1m: float,
    skip_llm: bool,
) -> tuple[dict, str | None]:
    if not candidates:
        result = skipped_llm_result(None)
        result["token_count_source"] = "no_candidates_no_llm_call"
        return result, None

    prompt = build_prompt(query, candidates)
    if skip_llm:
        return skipped_llm_result(prompt), prompt

    if client is None:
        raise RuntimeError("OpenAI client is required unless --skip-llm is used")

    return (
        call_orchestrator(
            client=client,
            model=model,
            prompt=prompt,
            candidates=candidates,
            input_price_per_1m=input_price_per_1m,
            output_price_per_1m=output_price_per_1m,
        ),
        prompt,
    )


def make_result_row(
    *,
    qid: str,
    query: str,
    positive_ids: set[str],
    n: int,
    method: str,
    k_budget: int,
    candidates: list[dict],
    retrieval_hit: int,
    offline_index_latency_s: float | None,
    retrieval_latency_s: float,
    query_signal_count: int | None,
    llm_result: dict,
    skip_llm: bool,
) -> dict:
    candidate_ids = {doc["pid"] for doc in candidates}
    selected_index = llm_result["selected_index"]
    selected_pid = llm_result["selected_pid"]
    valid_index = (
        isinstance(selected_index, int)
        and not isinstance(selected_index, bool)
        and 1 <= selected_index <= len(candidates)
    )
    valid_selection = (
        None
        if not llm_result["llm_attempted"]
        else int(
            llm_result["api_success"] == 1
            and valid_index
            and selected_pid in candidate_ids
        )
    )

    if skip_llm:
        end_to_end_correct = None
    else:
        end_to_end_correct = int(
            llm_result["api_success"] == 1 and selected_pid in positive_ids
        )

    conditional_selection_correct = None
    if (
        retrieval_hit == 1
        and llm_result["llm_attempted"] == 1
        and llm_result["api_success"] == 1
    ):
        conditional_selection_correct = int(selected_pid in positive_ids)

    online_latency_s = None
    if llm_result["llm_latency_s"] is not None:
        online_latency_s = retrieval_latency_s + llm_result["llm_latency_s"]

    return {
        "query_id": qid,
        "query": query,
        "positive_ids": sorted(positive_ids),
        "num_positive_ids": len(positive_ids),
        "N_agents": n,
        "method": method,
        "K_budget": k_budget,
        "pool_size": len(candidates),
        "candidate_recall": retrieval_hit,
        "conditional_selection_correct": conditional_selection_correct,
        "end_to_end_correct": end_to_end_correct,
        "valid_selection": valid_selection,
        "raw_llm_response": llm_result["raw_llm_response"],
        "selected_index": selected_index,
        "selection_parse_error": llm_result["selection_parse_error"],
        "selected_pid": selected_pid,
        "offline_index_latency_s": offline_index_latency_s,
        "retrieval_latency_s": retrieval_latency_s,
        "llm_latency_s": llm_result["llm_latency_s"],
        "online_end_to_end_latency_s": online_latency_s,
        "failure_latency_s": llm_result["failure_latency_s"],
        "query_signal_count": query_signal_count,
        "llm_attempted": llm_result["llm_attempted"],
        "api_success": llm_result["api_success"],
        "api_error": llm_result["api_error"],
        "input_tokens": llm_result["input_tokens"],
        "output_tokens": llm_result["output_tokens"],
        "token_count_source": llm_result["token_count_source"],
        "estimated_cost_usd": llm_result["estimated_cost_usd"],
    }


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["N_agents"], row["method"], row["K_budget"])].append(row)

    summary: list[dict] = []
    for (n, method, k_budget), items in grouped.items():
        attempted = [item for item in items if item["llm_attempted"] == 1]
        successful = [item for item in attempted if item["api_success"] == 1]
        conditional = [
            item["conditional_selection_correct"]
            for item in items
            if item["conditional_selection_correct"] is not None
        ]
        end_to_end = [
            item["end_to_end_correct"]
            for item in items
            if item["end_to_end_correct"] is not None
        ]
        valid = [
            item["valid_selection"]
            for item in successful
            if item["valid_selection"] is not None
        ]

        retrieval_latencies = [float(item["retrieval_latency_s"]) for item in items]
        llm_latencies = [
            float(item["llm_latency_s"])
            for item in successful
            if item["llm_latency_s"] is not None
        ]
        online_latencies = [
            float(item["online_end_to_end_latency_s"])
            for item in successful
            if item["online_end_to_end_latency_s"] is not None
        ]
        failure_latencies = [
            float(item["failure_latency_s"])
            for item in attempted
            if item["failure_latency_s"] is not None
        ]
        offline_latencies = [
            float(item["offline_index_latency_s"])
            for item in items
            if item["offline_index_latency_s"] is not None
        ]
        pool_sizes = [item["pool_size"] for item in items]
        input_tokens = [item["input_tokens"] for item in items]
        costs = [
            item["estimated_cost_usd"]
            for item in items
            if item["estimated_cost_usd"] is not None
        ]

        candidate_recall_rate = statistics.mean(item["candidate_recall"] for item in items)
        conditional_accuracy = statistics.mean(conditional) if conditional else None
        end_to_end_accuracy = statistics.mean(end_to_end) if end_to_end else None
        api_success_rate = (
            statistics.mean(item["api_success"] for item in attempted)
            if attempted
            else None
        )

        summary.append(
            {
                "N_agents": n,
                "Method": method,
                "K_budget": k_budget,
                "num_queries": len(items),
                "Candidate_recall": round(candidate_recall_rate, 4),
                "Conditional_selection_accuracy": round_or_none(conditional_accuracy, 4),
                "End_to_end_accuracy": round_or_none(end_to_end_accuracy, 4),
                "LLM_attempt_rate": round(len(attempted) / len(items), 4),
                "API_success_rate_attempted": round_or_none(api_success_rate, 4),
                "Valid_selection_rate_successful": round_or_none(
                    statistics.mean(valid) if valid else None, 4
                ),
                "Pool_size_mean": round(statistics.mean(pool_sizes), 2),
                "Pool_size_std": round(stdev_or_zero(pool_sizes), 2),
                "Pool_size_min": min(pool_sizes),
                "Pool_size_max": max(pool_sizes),
                "Prompt_tokens_mean": round(statistics.mean(input_tokens), 2),
                "Estimated_cost_usd_mean": round_or_none(mean_or_none(costs), 8),
                "Offline_index_latency_mean_s": round_or_none(
                    mean_or_none(offline_latencies), 6
                ),
                "Retrieval_latency_mean_s": round_or_none(
                    mean_or_none(retrieval_latencies), 6
                ),
                "Retrieval_latency_median_s": round_or_none(
                    median_or_none(retrieval_latencies), 6
                ),
                "Retrieval_latency_p95_s": round_or_none(
                    percentile(retrieval_latencies, 0.95), 6
                ),
                "LLM_latency_mean_success_s": round_or_none(
                    mean_or_none(llm_latencies), 6
                ),
                "LLM_latency_p95_success_s": round_or_none(
                    percentile(llm_latencies, 0.95), 6
                ),
                "Online_end_to_end_latency_mean_success_s": round_or_none(
                    mean_or_none(online_latencies), 6
                ),
                "Online_end_to_end_latency_p95_success_s": round_or_none(
                    percentile(online_latencies, 0.95), 6
                ),
                "Failure_latency_mean_s": round_or_none(
                    mean_or_none(failure_latencies), 6
                ),
                "API_failure_count": sum(
                    1 for item in attempted if item["api_success"] == 0
                ),
            }
        )

    summary.sort(key=lambda item: (item["N_agents"], item["Method"], item["K_budget"]))
    return summary


def write_outputs(
    output_dir: Path,
    rows: list[dict],
    summary: list[dict],
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    details_path = output_dir / "rq3_end_to_end_details.json"
    summary_path = output_dir / "rq3_end_to_end_summary.json"
    csv_path = output_dir / "rq3_end_to_end_summary.csv"
    metadata_path = output_dir / "rq3_experiment_metadata.json"

    with details_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    if summary:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    print("\nGenerated:")
    print(details_path)
    print(summary_path)
    print(csv_path)
    print(metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the corrected, bias-resistant RQ3 end-to-end benchmark."
    )
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION_PATH)
    parser.add_argument("--query-sample", type=Path, default=DEFAULT_QUERY_SAMPLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--n-values", type=parse_int_list, default=DEFAULT_N_VALUES)
    parser.add_argument("--k-values", type=parse_int_list, default=DEFAULT_K_VALUES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-negative-docs",
        type=int,
        default=DEFAULT_MAX_NEGATIVE_DOCS,
        help="Number of negative passages retained in the reproducible reservoir.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=DEFAULT_INPUT_PRICE_PER_1M,
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=DEFAULT_OUTPUT_PRICE_PER_1M,
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional deterministic prefix of the sampled queries for smoke tests.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Run registry/index/retrieval logic without making API calls.",
    )
    parser.add_argument(
        "--skip-traditional",
        action="store_true",
        help="Skip the full-catalog LLM baseline (useful for retrieval-only debugging).",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip the bounded BM25 retrieval baseline.",
    )
    parser.add_argument(
        "--skip-ssr",
        action="store_true",
        help="Skip the SSR bounded-retrieval method.",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    if args.max_negative_docs < max(args.n_values):
        raise ValueError(
            "--max-negative-docs must be at least max(--n-values) so every "
            "query can construct the largest registry."
        )

    sample_metadata, selected_queries = load_query_sample(args.query_sample)
    if args.max_queries is not None:
        if args.max_queries <= 0:
            raise ValueError("--max-queries must be positive")
        selected_queries = selected_queries[: args.max_queries]

    required_pids = {
        pid for item in selected_queries for pid in item["positive_ids"]
    }

    docs = load_documents_with_reservoir_negatives(
        collection_path=args.collection,
        required_pids=required_pids,
        max_negative_docs=args.max_negative_docs,
        negative_seed=args.seed,
        cache_path=args.cache,
    )

    print(f"Documents available to RQ3: {len(docs):,}")
    print(f"Queries evaluated: {len(selected_queries):,}")
    print(f"N values: {args.n_values}")
    print(f"K values: {args.k_values}")

    client = None
    if not args.skip_llm:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Use --skip-llm for a retrieval-only "
                "smoke test or configure the API key in the environment/.env."
            )
        from openai import OpenAI

        client = OpenAI()

    indexer = MSMarcoIndexer(max_signals_per_document=64, min_token_len=2)
    rows: list[dict] = []

    total_query_n_pairs = len(selected_queries) * len(args.n_values)
    completed_pairs = 0

    for query_index, item in enumerate(selected_queries, start=1):
        qid = item["query_id"]
        query = item["query"]
        positive_ids = set(item["positive_ids"])

        pools = build_nested_candidate_pools(
            qid=qid,
            positive_ids=item["positive_ids"],
            docs=docs,
            n_values=args.n_values,
            seed=args.seed,
        )

        for n in args.n_values:
            full_catalog = pools[n]

            # Build only the retrieval indexes enabled for this run. Their construction
            # times are recorded but excluded from online end-to-end latency.
            ssr_ranked = None
            ssr_offline_latency = None
            ssr_retrieval_latency = None
            ssr_query_signal_count = None
            if not args.skip_ssr:
                ssr_index, ssr_offline_latency = build_ssr_registry_index(
                    candidates=full_catalog,
                    indexer=indexer,
                )
                ssr_ranked, ssr_retrieval_latency, ssr_query_signal_count = retrieve_ssr_ranked(
                    query=query,
                    registry_index=ssr_index,
                    indexer=indexer,
                )

            bm25_index = None
            bm25_offline_latency = None
            if not args.skip_bm25:
                bm25_index, bm25_offline_latency = build_bm25_registry_index(
                    candidates=full_catalog,
                    indexer=indexer,
                )

            # Prepare all logical evaluations first, then use a deterministic
            # shuffled order for LLM calls to reduce systematic temporal bias.
            tasks: list[dict[str, Any]] = []

            if not args.skip_traditional:
                tasks.append(
                    {
                        "method": "Traditional",
                        "k_budget": n,
                        "candidates": full_catalog,
                        "retrieval_hit": 1,
                        "offline_index_latency_s": None,
                        "retrieval_latency_s": 0.0,
                        "query_signal_count": None,
                    }
                )

            for k in args.k_values:
                if ssr_ranked is not None:
                    ssr_candidates = ssr_ranked[:k]
                    ssr_candidate_ids = {doc["pid"] for doc in ssr_candidates}
                    tasks.append(
                        {
                            "method": "SSR",
                            "k_budget": k,
                            "candidates": ssr_candidates,
                            "retrieval_hit": int(bool(positive_ids.intersection(ssr_candidate_ids))),
                            "offline_index_latency_s": ssr_offline_latency,
                            "retrieval_latency_s": ssr_retrieval_latency,
                            "query_signal_count": ssr_query_signal_count,
                        }
                    )

                if bm25_index is not None:
                    bm25_candidates, bm25_retrieval_latency, bm25_query_term_count = (
                        retrieve_bm25_ranked(
                            query=query,
                            registry_index=bm25_index,
                            indexer=indexer,
                            top_k=k,
                        )
                    )
                    bm25_candidate_ids = {doc["pid"] for doc in bm25_candidates}
                    tasks.append(
                        {
                            "method": "BM25",
                            "k_budget": k,
                            "candidates": bm25_candidates,
                            "retrieval_hit": int(
                                bool(positive_ids.intersection(bm25_candidate_ids))
                            ),
                            "offline_index_latency_s": bm25_offline_latency,
                            "retrieval_latency_s": bm25_retrieval_latency,
                            "query_signal_count": bm25_query_term_count,
                        }
                    )

            if bm25_index is not None:
                bm25_index.connection.close()

            order_rng = random.Random(f"{args.seed}:{qid}:{n}:method-order")
            order_rng.shuffle(tasks)

            for task in tasks:
                llm_result, _ = evaluate_llm_pool(
                    query=query,
                    candidates=task["candidates"],
                    positive_ids=positive_ids,
                    client=client,
                    model=args.model,
                    input_price_per_1m=args.input_price_per_1m,
                    output_price_per_1m=args.output_price_per_1m,
                    skip_llm=args.skip_llm,
                )

                rows.append(
                    make_result_row(
                        qid=qid,
                        query=query,
                        positive_ids=positive_ids,
                        n=n,
                        method=task["method"],
                        k_budget=task["k_budget"],
                        candidates=task["candidates"],
                        retrieval_hit=task["retrieval_hit"],
                        offline_index_latency_s=task["offline_index_latency_s"],
                        retrieval_latency_s=task["retrieval_latency_s"],
                        query_signal_count=task["query_signal_count"],
                        llm_result=llm_result,
                        skip_llm=args.skip_llm,
                    )
                )

            completed_pairs += 1
            if completed_pairs % 20 == 0 or completed_pairs == total_query_n_pairs:
                print(
                    f"Progress: {completed_pairs}/{total_query_n_pairs} "
                    f"query-registry pairs ({completed_pairs / total_query_n_pairs:.1%})"
                )

    summary = summarize_rows(rows)

    metadata = {
        "experiment": "RQ3 corrected end-to-end infrastructure evaluation",
        "protocol_version": 4,
        "query_sample_path": str(args.query_sample),
        "query_sample_sha256": file_sha256(args.query_sample),
        "query_sample_metadata": sample_metadata,
        "queries_evaluated": len(selected_queries),
        "collection_path": str(args.collection),
        "collection_size_bytes": args.collection.stat().st_size,
        "n_values": args.n_values,
        "k_values": args.k_values,
        "seed": args.seed,
        "max_negative_docs": args.max_negative_docs,
        "negative_sampling": "reproducible reservoir sample from collection",
        "nested_registries": True,
        "nested_registry_definition": "A_N is built from one fixed per-query negative ordering",
        "ssr_candidate_policy": "top-K after IDF-weighted structural ranking",
        "bm25_candidate_policy": "top-K SQLite FTS5 bm25() over the same normalized unigram representation used in RQ2; no padding of non-matching passages",
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
        "full_catalog_fallback_on_empty_ssr": False,
        "full_catalog_fallback_on_empty_bm25": False,
        "llm_method_order": "deterministic shuffle per query-registry pair",
        "offline_indexing_excluded_from_online_latency": True,
        "indexer": {
            "class": "MSMarcoIndexer",
            "max_signals_per_document": 64,
            "min_token_len": 2,
        },
        "llm": {
            "enabled": not args.skip_llm,
            "model": args.model,
            "temperature": 0,
            "input_price_per_1m": args.input_price_per_1m,
            "output_price_per_1m": args.output_price_per_1m,
            "token_accounting": "API usage when available; character estimate fallback",
            "selection_output": "1-based selected_index",
            "selection_mapping": "selected_index is mapped to candidate PID internally",
            "selection_validation": "1 <= selected_index <= pool_size",
            "raw_response_saved": True,
        },
        "metrics": {
            "candidate_recall": "relevant passage appears in the bounded retrieval pool (SSR or BM25); full catalog is 1 by construction",
            "conditional_selection_accuracy": (
                "LLM selects a relevant passage among cases where retrieval hit and API call succeeded"
            ),
            "end_to_end_accuracy": (
                "final selected passage is relevant over all evaluated queries; retrieval misses and API failures count as incorrect"
            ),
            "online_end_to_end_latency": (
                "online retrieval (SSR or BM25, when applicable) + successful LLM inference; offline index construction excluded"
            ),
        },
        "skip_traditional": args.skip_traditional,
        "skip_ssr": args.skip_ssr,
        "skip_bm25": args.skip_bm25,
        "skip_llm": args.skip_llm,
    }

    write_outputs(args.output_dir, rows, summary, metadata)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()