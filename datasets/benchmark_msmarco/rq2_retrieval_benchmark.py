"""
RQ2 corrected retrieval benchmark for SSR on MS MARCO Passage Ranking.

This script adds classical lexical baselines and a component ablation without
modifying the legacy RQ2 scripts.

Methods
-------
1. TF-IDF (dot product): unigram lexical baseline, score = sum(qtf * dtf * idf).
2. BM25: SQLite FTS5 BM25 over the same normalized unigram representation.
3. Structural Overlap + ISI: unweighted SSR structural signals + inverted index.
4. Structural + IDF (full scan): same IDF score as SSR, but exhaustive scoring;
   by default this is evaluated on a deterministic latency sample because a
   full 8.8M x 6,980 exhaustive run is intentionally expensive.
5. SSR (Structural + IDF + ISI): complete proposed retrieval pipeline.

The TF-IDF baseline is intentionally reported as a dot-product TF-IDF baseline
(no cosine document-length normalization). BM25 provides the standard
length-normalized sparse lexical baseline.

Offline preprocessing/index construction is measured separately from online
query latency. All effectiveness metrics use the same qrels and eligible query
set. Full-scan IDF and SSR use the same scoring function, so their rankings are
identical; the full-scan variant isolates the computational contribution of ISI.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import heapq
import json
import math
import os
import platform
import random
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent
ROOT_DIR = BASE_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utilities.routing.msmarco_indexer import MSMarcoIndexer  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_COLLECTION = BASE_DIR / "collection.tsv"
DEFAULT_QUERIES = BASE_DIR / "queries.dev.small.tsv"
DEFAULT_QRELS = BASE_DIR / "qrels.dev.small.tsv"
DEFAULT_RESULTS_ROOT = BASE_DIR / "results" / "rq2_corrected"
DEFAULT_CACHE_DIR = BASE_DIR / "cache"

OFFICIAL_MAX_DOCS = 8_841_823
OFFICIAL_MAX_QUERIES = 6_980
DEFAULT_TOP_K = 10
DEFAULT_LATENCY_SAMPLE = 200
DEFAULT_SEED = 42

METHOD_TFIDF = "TF-IDF (dot product)"
METHOD_BM25 = "BM25"
METHOD_OVERLAP = "Structural Overlap + ISI"
METHOD_FULLSCAN = "Structural + IDF (full scan)"
METHOD_SSR = "SSR (Structural + IDF + ISI)"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GoldQuery:
    query_id: str
    query: str
    positive_ids: list[str]


@dataclass
class EvalSummary:
    method: str
    effectiveness_queries: int
    latency_queries: int
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    mrr_at_10: float | None
    latency_mean_ms: float | None
    latency_median_ms: float | None
    latency_p95_ms: float | None
    offline_time_ms: float | None
    avg_candidates: float | None
    avg_documents_scored: float | None
    notes: str


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def safe_round(value: float | None, digits: int = 6):
    return None if value is None else round(value, digits)


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


def parse_methods(value: str) -> list[str]:
    aliases = {
        "tfidf": METHOD_TFIDF,
        "bm25": METHOD_BM25,
        "overlap": METHOD_OVERLAP,
        "fullscan": METHOD_FULLSCAN,
        "ssr": METHOD_SSR,
    }
    result = []
    for token in value.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise argparse.ArgumentTypeError(
                f"Unknown method '{token}'. Use tfidf,bm25,overlap,fullscan,ssr."
            )
        result.append(aliases[key])
    if not result:
        raise argparse.ArgumentTypeError("At least one method is required.")
    return result


# ---------------------------------------------------------------------------
# MS MARCO loading
# ---------------------------------------------------------------------------
def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries[str(row[0])] = row[1].strip()
    return queries


def load_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            try:
                rel = int(row[3])
            except ValueError:
                continue
            if rel > 0:
                qrels[str(row[0])].append(str(row[2]))
    return dict(qrels)


def iter_collection(path: Path, max_docs: int | None) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        accepted = 0
        for row in reader:
            if len(row) < 2:
                continue
            if max_docs is not None and accepted >= max_docs:
                break
            accepted += 1
            yield {"id": str(row[0]), "text": row[1]}


def build_gold(
    queries: dict[str, str],
    qrels: dict[str, list[str]],
    valid_doc_ids: set[str] | None,
    max_queries: int | None,
) -> list[GoldQuery]:
    gold: list[GoldQuery] = []
    for qid, positives in qrels.items():
        query = queries.get(qid)
        if query is None:
            continue
        valid_positives = (
            list(positives)
            if valid_doc_ids is None
            else [pid for pid in positives if pid in valid_doc_ids]
        )
        if not valid_positives:
            continue
        gold.append(GoldQuery(qid, query, valid_positives))
        if max_queries is not None and len(gold) >= max_queries:
            break
    return gold


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def find_rank(ranked_ids: Sequence[str], positive_ids: Sequence[str]) -> int | None:
    positives = set(str(x) for x in positive_ids)
    for rank, pid in enumerate(ranked_ids, start=1):
        if str(pid) in positives:
            return rank
    return None


def metric_values(rank: int | None) -> dict[str, float]:
    return {
        "recall_at_1": 1.0 if rank is not None and rank <= 1 else 0.0,
        "recall_at_3": 1.0 if rank is not None and rank <= 3 else 0.0,
        "recall_at_5": 1.0 if rank is not None and rank <= 5 else 0.0,
        "recall_at_10": 1.0 if rank is not None and rank <= 10 else 0.0,
        "mrr_at_10": (1.0 / rank) if rank is not None and rank <= 10 else 0.0,
    }


def summarize_method(
    method: str,
    details: list[dict],
    offline_time_ms: float | None,
    notes: str,
    effectiveness_measured: bool = True,
    latency_query_ids: set[str] | None = None,
) -> EvalSummary:
    latency_rows = details
    if latency_query_ids is not None:
        latency_rows = [x for x in details if str(x.get("query_id")) in latency_query_ids]
    latencies = [float(x["latency_ms"]) for x in latency_rows if x.get("latency_ms") is not None]
    candidate_values = [float(x["candidate_count"]) for x in details if x.get("candidate_count") is not None]
    scored_values = [float(x["documents_scored"]) for x in details if x.get("documents_scored") is not None]

    if effectiveness_measured and details:
        r1 = statistics.fmean(x["recall_at_1"] for x in details)
        r3 = statistics.fmean(x["recall_at_3"] for x in details)
        r5 = statistics.fmean(x["recall_at_5"] for x in details)
        r10 = statistics.fmean(x["recall_at_10"] for x in details)
        mrr = statistics.fmean(x["mrr_at_10"] for x in details)
        eff_n = len(details)
    else:
        r1 = r3 = r5 = r10 = mrr = None
        eff_n = 0

    return EvalSummary(
        method=method,
        effectiveness_queries=eff_n,
        latency_queries=len(latencies),
        recall_at_1=safe_round(r1, 4),
        recall_at_3=safe_round(r3, 4),
        recall_at_5=safe_round(r5, 4),
        recall_at_10=safe_round(r10, 4),
        mrr_at_10=safe_round(mrr, 4),
        latency_mean_ms=safe_round(statistics.fmean(latencies) if latencies else None, 4),
        latency_median_ms=safe_round(statistics.median(latencies) if latencies else None, 4),
        latency_p95_ms=safe_round(percentile(latencies, 0.95), 4),
        offline_time_ms=safe_round(offline_time_ms, 4),
        avg_candidates=safe_round(statistics.fmean(candidate_values) if candidate_values else None, 4),
        avg_documents_scored=safe_round(statistics.fmean(scored_values) if scored_values else None, 4),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Structural variants
# ---------------------------------------------------------------------------
def build_structural_state(
    collection_path: Path,
    max_docs: int | None,
    log_interval: int,
):
    indexer = MSMarcoIndexer(max_signals_per_document=64, min_token_len=2)
    features = {}
    valid_doc_ids: set[str] = set()

    start_features = time.perf_counter()
    for i, doc in enumerate(iter_collection(collection_path, max_docs), start=1):
        feature = indexer.index_document(doc)
        pid = feature.agent_name
        features[pid] = feature
        valid_doc_ids.add(pid)
        if log_interval and i % log_interval == 0:
            elapsed = time.perf_counter() - start_features
            rate = i / elapsed if elapsed else 0.0
            print(f"[STRUCTURAL FEATURES] {i:,} docs | {rate:,.0f} docs/s")
    feature_time_ms = (time.perf_counter() - start_features) * 1000.0

    n_docs = len(features)
    print("Computing structural IDF...")
    start_idf = time.perf_counter()
    signal_df: dict[str, int] = defaultdict(int)
    for feature in features.values():
        for signal in feature.graph_signals:
            signal_df[signal] += 1
    signal_idf = {
        signal: math.log((n_docs + 1) / (df + 1)) + 1.0
        for signal, df in signal_df.items()
    }
    idf_time_ms = (time.perf_counter() - start_idf) * 1000.0

    print("Building structural Inverted Signal Index...")
    start_isi = time.perf_counter()
    signal_index: dict[str, set[str]] = defaultdict(set)
    for pid, feature in features.items():
        for signal in feature.graph_signals:
            signal_index[signal].add(pid)
    isi_time_ms = (time.perf_counter() - start_isi) * 1000.0

    timings = {
        "feature_extraction_ms": feature_time_ms,
        "idf_computation_ms": idf_time_ms,
        "isi_construction_ms": isi_time_ms,
        "total_ms": feature_time_ms + idf_time_ms + isi_time_ms,
    }
    return indexer, features, dict(signal_index), signal_idf, valid_doc_ids, timings


def topk_structural(
    query: str,
    indexer: MSMarcoIndexer,
    features: dict,
    signal_index: dict[str, set[str]],
    signal_idf: dict[str, float],
    top_k: int,
    weighted: bool,
    use_isi: bool,
):
    query_signals = indexer.extract_query_signals(query)

    if use_isi:
        candidate_ids: set[str] = set()
        for signal in query_signals:
            candidate_ids.update(signal_index.get(signal, set()))
        iterator = ((pid, features[pid]) for pid in candidate_ids if pid in features)
        documents_scored = len(candidate_ids)
    else:
        candidate_ids = set()
        iterator = features.items()
        documents_scored = len(features)

    heap: list[tuple[tuple[float, int, str], str, list[str]]] = []
    candidate_count = 0

    for pid, feature in iterator:
        matched = query_signals.intersection(feature.graph_signals)
        if not matched:
            continue
        candidate_count += 1
        if weighted:
            score = sum(signal_idf.get(s, 1.0) for s in matched)
        else:
            score = float(len(matched))
        key = (score, len(matched), str(pid))
        payload = (key, str(pid), sorted(matched))
        if len(heap) < top_k:
            heapq.heappush(heap, payload)
        elif key > heap[0][0]:
            heapq.heapreplace(heap, payload)

    ranked = sorted(heap, key=lambda x: x[0], reverse=True)
    ranked_ids = [pid for _, pid, _ in ranked]
    return ranked_ids, candidate_count, documents_scored


def evaluate_structural_method(
    method: str,
    gold: Sequence[GoldQuery],
    indexer: MSMarcoIndexer,
    features: dict,
    signal_index: dict[str, set[str]],
    signal_idf: dict[str, float],
    top_k: int,
    log_interval: int,
    weighted: bool,
    use_isi: bool,
) -> list[dict]:
    details: list[dict] = []
    for i, item in enumerate(gold, start=1):
        start = time.perf_counter()
        ranked_ids, candidate_count, documents_scored = topk_structural(
            query=item.query,
            indexer=indexer,
            features=features,
            signal_index=signal_index,
            signal_idf=signal_idf,
            top_k=top_k,
            weighted=weighted,
            use_isi=use_isi,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        rank = find_rank(ranked_ids, item.positive_ids)
        metrics = metric_values(rank)
        details.append({
            "method": method,
            "query_id": item.query_id,
            "query": item.query,
            "positive_ids": item.positive_ids,
            "rank": rank,
            **metrics,
            "latency_ms": round(latency_ms, 6),
            "candidate_count": candidate_count,
            "documents_scored": documents_scored,
            "top_document_ids": ranked_ids,
        })
        if log_interval and i % log_interval == 0:
            avg = statistics.fmean(x["latency_ms"] for x in details)
            print(f"[{method}] {i:,}/{len(gold):,} queries | avg={avg:,.2f} ms/query")
    return details


# ---------------------------------------------------------------------------
# SQLite FTS5 lexical baselines
# ---------------------------------------------------------------------------
def lexical_db_paths(cache_dir: Path, max_docs: int | None) -> tuple[Path, Path]:
    tag = "all" if max_docs is None else str(max_docs)
    return (
        cache_dir / f"rq2_lexical_{tag}.sqlite3",
        cache_dir / f"rq2_lexical_{tag}.meta.json",
    )


def lexical_cache_signature(collection_path: Path, max_docs: int | None) -> dict:
    st = collection_path.stat()
    return {
        "collection_path": str(collection_path.resolve()),
        "collection_size_bytes": st.st_size,
        "collection_mtime_ns": st.st_mtime_ns,
        "max_docs": max_docs,
        "tokenizer": "MSMarcoIndexer._tokenize(min_token_len=2, built-in stopwords)",
    }


def ensure_fts5(con: sqlite3.Connection):
    try:
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_test USING fts5(x)")
        con.execute("DROP TABLE IF EXISTS __fts5_test")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "This Python/SQLite build does not provide FTS5, required for the BM25/TF-IDF baselines."
        ) from exc


def build_or_open_lexical_index(
    collection_path: Path,
    max_docs: int | None,
    cache_dir: Path,
    log_interval: int,
    force_rebuild: bool,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path, meta_path = lexical_db_paths(cache_dir, max_docs)
    expected = lexical_cache_signature(collection_path, max_docs)

    cache_valid = False
    if not force_rebuild and db_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cache_valid = all(meta.get(k) == v for k, v in expected.items()) and meta.get("complete") is True
        except Exception:
            cache_valid = False

    if cache_valid:
        print(f"Loading validated lexical FTS5 index: {db_path}")
        con = sqlite3.connect(str(db_path))
        con.execute("PRAGMA query_only=ON")
        n_docs = int(json.loads(meta_path.read_text(encoding="utf-8"))["document_count"])
        return con, db_path, meta_path, 0.0, n_docs, True

    if db_path.exists():
        db_path.unlink()
    if meta_path.exists():
        meta_path.unlink()

    print(f"Building lexical FTS5 index: {db_path}")
    con = sqlite3.connect(str(db_path))
    ensure_fts5(con)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-262144")
    con.execute("CREATE VIRTUAL TABLE docs USING fts5(pid UNINDEXED, terms, tokenize='unicode61 remove_diacritics 0')")

    indexer = MSMarcoIndexer(max_signals_per_document=64, min_token_len=2)
    start = time.perf_counter()
    batch: list[tuple[str, str]] = []
    n_docs = 0

    for doc in iter_collection(collection_path, max_docs):
        # Baselines use normalized unigrams only; no bigram structural signals.
        tokens = indexer._tokenize(doc["text"])
        batch.append((str(doc["id"]), " ".join(tokens)))
        n_docs += 1
        if len(batch) >= 20_000:
            con.executemany("INSERT INTO docs(pid, terms) VALUES (?, ?)", batch)
            con.commit()
            batch.clear()
        if log_interval and n_docs % log_interval == 0:
            elapsed = time.perf_counter() - start
            rate = n_docs / elapsed if elapsed else 0.0
            print(f"[LEXICAL INDEX] {n_docs:,} docs | {rate:,.0f} docs/s")

    if batch:
        con.executemany("INSERT INTO docs(pid, terms) VALUES (?, ?)", batch)
        con.commit()

    print("Optimizing FTS5 index...")
    con.execute("INSERT INTO docs(docs) VALUES ('optimize')")
    con.commit()
    con.execute("CREATE VIRTUAL TABLE vocab_row USING fts5vocab(docs, 'row')")
    con.execute("CREATE VIRTUAL TABLE vocab_instance USING fts5vocab(docs, 'instance')")
    con.commit()

    offline_ms = (time.perf_counter() - start) * 1000.0
    meta = {
        **expected,
        "complete": True,
        "document_count": n_docs,
        "offline_index_time_ms": round(offline_ms, 4),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Query connection configuration.
    con.execute("PRAGMA cache_size=-262144")
    return con, db_path, meta_path, offline_ms, n_docs, False


def baseline_query_tokens(query: str) -> list[str]:
    indexer = MSMarcoIndexer(max_signals_per_document=64, min_token_len=2)
    return indexer._tokenize(query)


def fts_quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def rank_bm25(con: sqlite3.Connection, query: str, top_k: int):
    tokens = baseline_query_tokens(query)
    if not tokens:
        return [], 0
    unique = list(dict.fromkeys(tokens))
    match_expr = " OR ".join(fts_quote(t) for t in unique)
    rows = con.execute(
        "SELECT pid, bm25(docs) AS score FROM docs "
        "WHERE docs MATCH ? ORDER BY score ASC, pid ASC LIMIT ?",
        (match_expr, top_k),
    ).fetchall()
    # Candidate count is intentionally omitted because COUNT(*) over every
    # matching posting can materially distort measured query latency.
    return [str(r[0]) for r in rows], None


def rank_tfidf_dot(con: sqlite3.Connection, query: str, top_k: int, n_docs: int):
    tokens = baseline_query_tokens(query)
    if not tokens:
        return [], 0
    qtf = Counter(tokens)
    terms = list(qtf.keys())
    placeholders = ",".join("?" for _ in terms)
    df_rows = con.execute(
        f"SELECT term, doc FROM vocab_row WHERE term IN ({placeholders})",
        terms,
    ).fetchall()
    idf = {
        str(term): math.log((n_docs + 1) / (int(df) + 1)) + 1.0
        for term, df in df_rows
    }
    active_terms = [t for t in terms if t in idf]
    if not active_terms:
        return [], 0

    values_sql = ",".join("(?,?,?)" for _ in active_terms)
    params: list[object] = []
    for term in active_terms:
        params.extend([term, int(qtf[term]), float(idf[term])])
    params.append(top_k)

    sql = f"""
        WITH qterms(term, qtf, idf) AS (VALUES {values_sql})
        SELECT d.pid, SUM(q.qtf * q.idf) AS score
        FROM vocab_instance AS v
        JOIN qterms AS q ON q.term = v.term
        JOIN docs AS d ON d.rowid = v.doc
        GROUP BY v.doc
        ORDER BY score DESC, d.pid ASC
        LIMIT ?
    """
    rows = con.execute(sql, params).fetchall()
    return [str(r[0]) for r in rows], None


def evaluate_lexical_method(
    method: str,
    gold: Sequence[GoldQuery],
    con: sqlite3.Connection,
    n_docs: int,
    top_k: int,
    log_interval: int,
) -> list[dict]:
    details: list[dict] = []
    for i, item in enumerate(gold, start=1):
        start = time.perf_counter()
        if method == METHOD_BM25:
            ranked_ids, candidate_count = rank_bm25(con, item.query, top_k)
        elif method == METHOD_TFIDF:
            ranked_ids, candidate_count = rank_tfidf_dot(con, item.query, top_k, n_docs)
        else:
            raise ValueError(method)
        latency_ms = (time.perf_counter() - start) * 1000.0
        rank = find_rank(ranked_ids, item.positive_ids)
        metrics = metric_values(rank)
        details.append({
            "method": method,
            "query_id": item.query_id,
            "query": item.query,
            "positive_ids": item.positive_ids,
            "rank": rank,
            **metrics,
            "latency_ms": round(latency_ms, 6),
            "candidate_count": candidate_count,
            "documents_scored": None,
            "top_document_ids": ranked_ids,
        })
        if log_interval and i % log_interval == 0:
            avg = statistics.fmean(x["latency_ms"] for x in details)
            print(f"[{method}] {i:,}/{len(gold):,} queries | avg={avg:,.2f} ms/query")
    return details


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_method_checkpoint(run_dir: Path, method: str, details: list[dict], summary: EvalSummary):
    run_dir.mkdir(parents=True, exist_ok=True)
    safe = (
        method.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    (run_dir / f"checkpoint_{safe}_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    (run_dir / f"checkpoint_{safe}_summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    print(f"Checkpoint saved for {method}.")


def write_outputs(
    run_dir: Path,
    summaries: list[EvalSummary],
    details_by_method: dict[str, list[dict]],
    metadata: dict,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_json = run_dir / "rq2_summary.json"
    summary_csv = run_dir / "rq2_summary.csv"
    details_json = run_dir / "rq2_details.json"
    metadata_json = run_dir / "rq2_experiment_metadata.json"

    summary_payload = [asdict(x) for x in summaries]
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    details_json.write_text(json.dumps(details_by_method, indent=2), encoding="utf-8")
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fields = list(asdict(summaries[0]).keys()) if summaries else [
        "method", "effectiveness_queries", "latency_queries", "recall_at_1",
        "recall_at_3", "recall_at_5", "recall_at_10", "mrr_at_10",
        "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
        "offline_time_ms", "avg_candidates", "avg_documents_scored", "notes",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_payload:
            writer.writerow(row)

    print("\nGenerated:")
    for path in (summary_json, summary_csv, details_json, metadata_json):
        print(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Corrected RQ2 retrieval benchmark")
    p.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    p.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--run-name", default=None, help="Output subdirectory; default is a UTC timestamp.")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--latency-sample", type=int, default=DEFAULT_LATENCY_SAMPLE,
                   help="Number of deterministic queries used for full-scan latency ablation.")
    p.add_argument("--max-docs", type=int, default=100_000,
                   help="Smoke-test corpus cap. Use --official for the full corpus.")
    p.add_argument("--max-queries", type=int, default=200,
                   help="Smoke-test query cap. Use --official for the full dev-small set.")
    p.add_argument("--official", action="store_true",
                   help=f"Use up to {OFFICIAL_MAX_DOCS:,} docs and {OFFICIAL_MAX_QUERIES:,} queries.")
    p.add_argument("--methods", type=parse_methods,
                   default=parse_methods("tfidf,bm25,overlap,fullscan,ssr"),
                   help="Comma-separated: tfidf,bm25,overlap,fullscan,ssr")
    p.add_argument("--full-scan-all", action="store_true",
                   help="Evaluate exhaustive Structural+IDF on all eligible queries. Very expensive on full MS MARCO.")
    p.add_argument("--force-lexical-rebuild", action="store_true")
    p.add_argument("--index-log-interval", type=int, default=100_000)
    p.add_argument("--query-log-interval", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    if args.official:
        max_docs: int | None = OFFICIAL_MAX_DOCS
        max_queries: int | None = OFFICIAL_MAX_QUERIES
    else:
        max_docs = args.max_docs
        max_queries = args.max_queries

    methods: list[str] = args.methods
    run_name = args.run_name or now_run_id()
    run_dir = args.results_root / run_name

    for path in (args.collection, args.queries, args.qrels):
        if not path.exists():
            raise FileNotFoundError(path)

    print("===== RQ2 CORRECTED BENCHMARK =====")
    print(f"Collection: {args.collection}")
    print(f"Max docs: {max_docs if max_docs is not None else 'ALL'}")
    print(f"Max queries: {max_queries if max_queries is not None else 'ALL'}")
    print(f"Top K: {args.top_k}")
    print(f"Methods: {methods}")
    print(f"Run directory: {run_dir}")

    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)

    summaries: list[EvalSummary] = []
    details_by_method: dict[str, list[dict]] = {}
    gold: list[GoldQuery] | None = None
    valid_doc_ids: set[str] | None = None

    structural_methods = {METHOD_OVERLAP, METHOD_FULLSCAN, METHOD_SSR}
    need_structural = any(m in structural_methods for m in methods)

    structural_timings = None
    n_structural_signals = None
    latency_query_ids: set[str] = set()

    if need_structural:
        print("\nBuilding structural representation, IDF, and ISI...")
        (
            indexer,
            features,
            signal_index,
            signal_idf,
            valid_doc_ids,
            structural_timings,
        ) = build_structural_state(
            args.collection, max_docs, args.index_log_interval
        )
        n_structural_signals = len(signal_index)
        gold = build_gold(queries, qrels, valid_doc_ids, max_queries)
        print(f"Structural documents: {len(features):,}")
        print(f"Eligible queries: {len(gold):,}")
        print(f"Structural signals: {n_structural_signals:,}")
        print(f"Structural offline timings: {json.dumps(structural_timings, indent=2)}")

        rng = random.Random(args.seed)
        sample_size = min(args.latency_sample, len(gold))
        latency_indices = sorted(rng.sample(range(len(gold)), sample_size))
        latency_query_ids = {gold[i].query_id for i in latency_indices}
        latency_gold = [gold[i] for i in latency_indices]
        print(f"Common online-latency sample: {len(latency_gold):,} queries (seed={args.seed})")

        overlap_offline_ms = (
            structural_timings["feature_extraction_ms"]
            + structural_timings["isi_construction_ms"]
        )
        fullscan_offline_ms = (
            structural_timings["feature_extraction_ms"]
            + structural_timings["idf_computation_ms"]
        )
        ssr_offline_ms = structural_timings["total_ms"]

        if METHOD_OVERLAP in methods:
            print(f"\nRunning {METHOD_OVERLAP}...")
            d = evaluate_structural_method(
                METHOD_OVERLAP, gold, indexer, features, signal_index, signal_idf,
                args.top_k, args.query_log_interval, weighted=False, use_isi=True,
            )
            details_by_method[METHOD_OVERLAP] = d
            summary = summarize_method(
                METHOD_OVERLAP, d, overlap_offline_ms,
                "Unweighted structural-signal overlap; ISI restricts scoring to matching documents.",
                latency_query_ids=latency_query_ids,
            )
            summaries.append(summary)
            write_method_checkpoint(run_dir, METHOD_OVERLAP, d, summary)

        if METHOD_SSR in methods:
            print(f"\nRunning {METHOD_SSR}...")
            d = evaluate_structural_method(
                METHOD_SSR, gold, indexer, features, signal_index, signal_idf,
                args.top_k, args.query_log_interval, weighted=True, use_isi=True,
            )
            details_by_method[METHOD_SSR] = d
            summary = summarize_method(
                METHOD_SSR, d, ssr_offline_ms,
                "Complete SSR: IDF-weighted structural ranking restricted by ISI.",
                latency_query_ids=latency_query_ids,
            )
            summaries.append(summary)
            write_method_checkpoint(run_dir, METHOD_SSR, d, summary)

        if METHOD_FULLSCAN in methods:
            if args.full_scan_all:
                fullscan_gold = list(gold)
                note = (
                    "Exhaustive IDF-weighted scoring over the complete corpus; "
                    "all eligible queries evaluated directly."
                )
            else:
                fullscan_gold = list(latency_gold)
                note = (
                    f"Exhaustive IDF-weighted scoring on the common deterministic {len(fullscan_gold)}-query "
                    f"latency sample (seed={args.seed}). Its scoring function is identical to SSR; "
                    f"only candidate enumeration differs."
                )

            print(f"\nRunning {METHOD_FULLSCAN} on {len(fullscan_gold):,} queries...")
            d = evaluate_structural_method(
                METHOD_FULLSCAN, fullscan_gold, indexer, features, signal_index,
                signal_idf, args.top_k, args.query_log_interval,
                weighted=True, use_isi=False,
            )
            details_by_method[METHOD_FULLSCAN] = d

            # First compute the full-scan latency/candidate statistics from the
            # queries that were actually executed. Effectiveness is either
            # measured directly (--full-scan-all) or, in the default protocol,
            # reported from SSR only after ranking equivalence is verified.
            summary = summarize_method(
                METHOD_FULLSCAN, d, fullscan_offline_ms, note,
                effectiveness_measured=args.full_scan_all,
                latency_query_ids=latency_query_ids,
            )

            if METHOD_SSR in details_by_method:
                ssr_map = {
                    x["query_id"]: x["top_document_ids"]
                    for x in details_by_method[METHOD_SSR]
                }
                mismatches = [
                    x["query_id"] for x in d
                    if ssr_map.get(x["query_id"]) != x["top_document_ids"]
                ]
                if mismatches:
                    raise RuntimeError(
                        "Full-scan IDF and SSR rankings differ for query IDs: "
                        + ", ".join(mismatches[:10])
                    )

                verification_scope = (
                    "all eligible queries" if args.full_scan_all
                    else f"the deterministic {len(fullscan_gold)}-query latency sample"
                )
                print(
                    "Verified: full-scan IDF and SSR rankings are identical on "
                    + verification_scope
                    + "."
                )

                # When the exhaustive variant is intentionally run only on the
                # latency sample, do not leave its paper-facing effectiveness
                # cells empty. Full-scan IDF and SSR have the same score for
                # every document with non-zero structural overlap; ISI merely
                # enumerates that same non-zero candidate set. After the
                # empirical equivalence check above, report the already-measured
                # SSR effectiveness values for this ablation row while keeping
                # the full-scan latency measured from its own exhaustive calls.
                if not args.full_scan_all:
                    ssr_summary = next(
                        (x for x in summaries if x.method == METHOD_SSR), None
                    )
                    if ssr_summary is None:
                        raise RuntimeError(
                            "SSR summary is required to report full-scan effectiveness "
                            "without --full-scan-all."
                        )

                    summary.effectiveness_queries = ssr_summary.effectiveness_queries
                    summary.recall_at_1 = ssr_summary.recall_at_1
                    summary.recall_at_3 = ssr_summary.recall_at_3
                    summary.recall_at_5 = ssr_summary.recall_at_5
                    summary.recall_at_10 = ssr_summary.recall_at_10
                    summary.mrr_at_10 = ssr_summary.mrr_at_10
                    summary.notes = (
                        note
                        + " Effectiveness metrics are reported from SSR after "
                        + "verifying ranking equivalence on the common latency sample; "
                        + "the exhaustive variant is not redundantly executed on all "
                        + "effectiveness queries."
                    )
                    print(
                        "Full-scan effectiveness metrics populated from SSR after "
                        "verified ranking equivalence."
                    )
            elif not args.full_scan_all:
                print(
                    "Warning: SSR was not requested, so full-scan effectiveness "
                    "metrics remain unset. Include 'ssr' in --methods or use "
                    "--full-scan-all to measure them directly."
                )

            summaries.append(summary)
            write_method_checkpoint(run_dir, METHOD_FULLSCAN, d, summary)

        # Release the very large Python structural state before opening/building
        # the disk-backed lexical index.
        del features, signal_index, signal_idf
        gc.collect()

    lexical_methods = {METHOD_TFIDF, METHOD_BM25}
    if any(m in lexical_methods for m in methods):
        print("\nPreparing lexical baseline index...")
        con, db_path, meta_path, lexical_offline_ms, lexical_n_docs, reused = build_or_open_lexical_index(
            args.collection,
            max_docs,
            args.cache_dir,
            args.index_log_interval,
            args.force_lexical_rebuild,
        )
        print(f"Lexical documents: {lexical_n_docs:,}")
        print(f"Lexical index reused: {reused}")

        # If no structural method was requested, derive valid IDs directly from
        # the FTS table. This is memory-heavy for the full collection; normal
        # all-method runs reuse the already-derived structural gold set.
        if gold is None:
            if max_docs is None or max_docs >= OFFICIAL_MAX_DOCS:
                # The official index covers the complete collection, so all qrel
                # document IDs can be used directly without materializing 8.8M IDs.
                gold = build_gold(queries, qrels, None, max_queries)
            else:
                print("Loading valid document IDs from lexical index for gold filtering...")
                valid_doc_ids = {str(row[0]) for row in con.execute("SELECT pid FROM docs")}
                gold = build_gold(queries, qrels, valid_doc_ids, max_queries)
                del valid_doc_ids
                gc.collect()
        print(f"Eligible queries for lexical baselines: {len(gold):,}")

        if METHOD_TFIDF in methods:
            print(f"\nRunning {METHOD_TFIDF}...")
            d = evaluate_lexical_method(
                METHOD_TFIDF, gold, con, lexical_n_docs, args.top_k, args.query_log_interval
            )
            details_by_method[METHOD_TFIDF] = d
            summary = summarize_method(
                METHOD_TFIDF, d, lexical_offline_ms if not reused else 0.0,
                "Classical unigram TF-IDF dot-product baseline; raw term frequency and smoothed IDF, without cosine normalization.",
                latency_query_ids=latency_query_ids if latency_query_ids else None,
            )
            summaries.append(summary)
            write_method_checkpoint(run_dir, METHOD_TFIDF, d, summary)

        if METHOD_BM25 in methods:
            print(f"\nRunning {METHOD_BM25}...")
            d = evaluate_lexical_method(
                METHOD_BM25, gold, con, lexical_n_docs, args.top_k, args.query_log_interval
            )
            details_by_method[METHOD_BM25] = d
            summary = summarize_method(
                METHOD_BM25, d, lexical_offline_ms if not reused else 0.0,
                "SQLite FTS5 BM25 over the same normalized unigram representation used by the lexical baseline.",
                latency_query_ids=latency_query_ids if latency_query_ids else None,
            )
            summaries.append(summary)
            write_method_checkpoint(run_dir, METHOD_BM25, d, summary)
        con.close()

    # Order rows for paper readability.
    order = [METHOD_TFIDF, METHOD_BM25, METHOD_OVERLAP, METHOD_FULLSCAN, METHOD_SSR]
    summaries.sort(key=lambda x: order.index(x.method) if x.method in order else 999)

    metadata = {
        "protocol_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MS MARCO Passage Ranking Dev Small",
        "collection_path": str(args.collection.resolve()),
        "queries_path": str(args.queries.resolve()),
        "qrels_path": str(args.qrels.resolve()),
        "queries_sha256": sha256_file(args.queries),
        "qrels_sha256": sha256_file(args.qrels),
        "collection_size_bytes": args.collection.stat().st_size,
        "collection_mtime_ns": args.collection.stat().st_mtime_ns,
        "max_docs": max_docs,
        "max_queries": max_queries,
        "eligible_queries": len(gold) if gold is not None else 0,
        "top_k": args.top_k,
        "seed": args.seed,
        "full_scan_latency_sample": args.latency_sample,
        "full_scan_all": args.full_scan_all,
        "full_scan_effectiveness_reporting": (
            "direct exhaustive evaluation on all eligible queries"
            if args.full_scan_all
            else (
                "SSR effectiveness after empirical ranking-equivalence verification "
                "on the deterministic latency sample; full-scan latency measured directly"
                if METHOD_FULLSCAN in methods and METHOD_SSR in methods
                else "not populated unless measured directly"
            )
        ),
        "common_latency_query_count": len(latency_query_ids),
        "common_latency_query_ids": sorted(latency_query_ids),
        "structural_offline_timings_ms": structural_timings,
        "methods": methods,
        "structural_extraction": {
            "max_signals_per_document": 64,
            "min_token_len": 2,
            "signals": "normalized unigrams followed by consecutive bigrams",
        },
        "idf_formula": "log((N+1)/(df+1)) + 1",
        "tfidf_formula": "sum_t qtf(t) * dtf(t,d) * idf(t); no cosine normalization",
        "bm25_implementation": "SQLite FTS5 bm25()",
        "latency_scope": "online query execution only; offline index construction reported separately",
        "python_version": sys.version,
        "platform": platform.platform(),
        "script": str(THIS_FILE),
        "run_name": run_name,
    }

    write_outputs(run_dir, summaries, details_by_method, metadata)

    print("\n===== SUMMARY =====")
    print(json.dumps([asdict(x) for x in summaries], indent=2))


if __name__ == "__main__":
    main()
