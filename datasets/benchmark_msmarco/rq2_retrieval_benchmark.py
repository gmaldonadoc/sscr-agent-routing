"""
RQ2 retrieval benchmark for SSR on MS MARCO Passage Ranking.

Protocol v3 is designed for the full 8,841,823-passage experiment and the RQ2
ablation without materializing the complete structural state in Python RAM.

A single persistent SQLite cache is built in batches. During one pass over the
collection, each passage is tokenized once and used to populate:
  * a disk-backed document table containing compact structural signals;
  * a structural FTS5 inverted index (the ISI implementation);
  * a lexical FTS5 index used by TF-IDF and BM25.

Methods
-------
1. TF-IDF (dot product): unigram lexical baseline, score = sum(qtf * dtf * idf).
2. BM25: SQLite FTS5 BM25 over the same normalized unigram representation.
3. Structural Overlap + ISI: unweighted structural signals + inverted index.
4. Structural + IDF (full scan): same structural-IDF scoring as SSR, but every
   document is visited. By default this is run only on a deterministic sample.
5. SSR (Structural + IDF + ISI): structural IDF ranking restricted by ISI.

Effectiveness is evaluated on the complete eligible query set for TF-IDF, BM25,
Structural Overlap + ISI, and SSR. The full-scan ablation uses the same score as
SSR. When its top-k output is verified to be identical to SSR on the deterministic
sample, the script reports SSR's effectiveness values for the full-scan row and
marks them as analytically equivalent rather than re-running 8.8M x 6,980 scans.

Offline index construction is reported separately from online query latency.
If a validated cache is reused, the original construction time stored in cache
metadata is retained; it is never replaced by zero.
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
import pickle
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
from typing import Callable, Iterable, Sequence

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
# Defaults / protocol constants
# ---------------------------------------------------------------------------
PROTOCOL_VERSION = 3
INDEX_SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "msmarco_structural_v1_unigrams_bigrams_64"

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
DEFAULT_INDEX_BATCH_SIZE = 25_000
DEFAULT_FULLSCAN_FETCH_SIZE = 10_000
DEFAULT_SQLITE_CACHE_MB = 512

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


def quick_file_fingerprint(path: Path, sample_bytes: int = 1024 * 1024) -> dict:
    """
    Fast collection identity check without hashing the entire multi-GB corpus.
    Uses file size plus SHA-256 over the first and last 1 MiB.
    """
    st = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        first = f.read(sample_bytes)
        h.update(first)
        if st.st_size > sample_bytes:
            f.seek(max(0, st.st_size - sample_bytes))
            h.update(f.read(sample_bytes))
    h.update(str(st.st_size).encode("ascii"))
    return {
        "size_bytes": st.st_size,
        "first_last_sha256": h.hexdigest(),
    }


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
    result: list[str] = []
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


def db_size_bytes(path: Path) -> int:
    total = path.stat().st_size if path.exists() else 0
    for suffix in ("-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            total += p.stat().st_size
    return total


def format_gib(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GiB"


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


def iter_collection(
    path: Path,
    max_docs: int | None,
    start_after: int = 0,
) -> Iterable[dict[str, str]]:
    """Yield valid collection rows, optionally skipping already-indexed rows."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        accepted = 0
        for row in reader:
            if len(row) < 2:
                continue
            if max_docs is not None and accepted >= max_docs:
                break
            accepted += 1
            if accepted <= start_after:
                continue
            yield {"id": str(row[0]), "text": row[1], "ordinal": accepted}


def build_gold(
    queries: dict[str, str],
    qrels: dict[str, list[str]],
    max_queries: int | None,
    doc_exists: Callable[[str], bool] | None = None,
) -> list[GoldQuery]:
    gold: list[GoldQuery] = []
    for qid, positives in qrels.items():
        query = queries.get(qid)
        if query is None:
            continue
        if doc_exists is None:
            valid_positives = list(positives)
        else:
            valid_positives = [pid for pid in positives if doc_exists(pid)]
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


def copy_effectiveness(source: EvalSummary, target: EvalSummary, total_queries: int) -> None:
    target.effectiveness_queries = total_queries
    target.recall_at_1 = source.recall_at_1
    target.recall_at_3 = source.recall_at_3
    target.recall_at_5 = source.recall_at_5
    target.recall_at_10 = source.recall_at_10
    target.mrr_at_10 = source.mrr_at_10


# ---------------------------------------------------------------------------
# Unified disk-backed RQ2 index
# ---------------------------------------------------------------------------
def unified_index_paths(cache_dir: Path, max_docs: int | None) -> tuple[Path, Path]:
    tag = "all" if max_docs is None else str(max_docs)
    return (
        cache_dir / f"rq2_unified_{tag}.sqlite3",
        cache_dir / f"rq2_unified_{tag}.meta.json",
    )


def unified_cache_signature(collection_path: Path, max_docs: int | None) -> dict:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "collection_fingerprint": quick_file_fingerprint(collection_path),
        "max_docs": max_docs,
        "extractor_version": EXTRACTOR_VERSION,
        "max_signals_per_document": 64,
        "min_token_len": 2,
        "idf_formula": "log((N+1)/(df+1)) + 1",
        "structural_tokenizer": "lowercase [a-z0-9]+, min_len=2, built-in stopwords, unigrams+bigrams",
        "lexical_representation": "same normalized unigrams with term multiplicity preserved",
    }


def ensure_fts5(con: sqlite3.Connection) -> None:
    try:
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_test USING fts5(x)")
        con.execute("DROP TABLE IF EXISTS __fts5_test")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "This Python/SQLite build does not provide FTS5, required by RQ2."
        ) from exc


def configure_build_connection(
    con: sqlite3.Connection,
    cache_mb: int,
    unsafe_fast_build: bool,
) -> None:
    ensure_fts5(con)
    if unsafe_fast_build:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
    else:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA wal_autocheckpoint=10000")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute(f"PRAGMA cache_size=-{max(64, cache_mb) * 1024}")
    con.execute("PRAGMA mmap_size=1073741824")


def configure_query_connection(con: sqlite3.Connection, cache_mb: int) -> None:
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute(f"PRAGMA cache_size=-{max(64, cache_mb) * 1024}")
    con.execute("PRAGMA mmap_size=2147483648")


def create_unified_schema(con: sqlite3.Connection) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS docs("
        "rowid INTEGER PRIMARY KEY, "
        "pid TEXT NOT NULL UNIQUE, "
        "signals TEXT NOT NULL)"
    )
    # Structural signals are unique per document. Underscore must remain part of
    # a token so a bigram such as 'reserve_bank' is one structural signal.
    con.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS structural_fts USING fts5("
        "signals, content='', columnsize=0, "
        "tokenize=\"unicode61 remove_diacritics 0 tokenchars '_'\")"
    )
    # Lexical FTS is contentless but keeps document-size information required by
    # BM25. Duplicate unigrams are preserved, which also provides dtf for TF-IDF.
    con.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts USING fts5("
        "terms, content='', tokenize='unicode61 remove_diacritics 0')"
    )
    con.commit()


def fresh_timing_dict() -> dict[str, float]:
    return {
        "tokenization_ms": 0.0,
        "signal_extraction_ms": 0.0,
        "docs_insert_ms": 0.0,
        "structural_fts_insert_ms": 0.0,
        "lexical_fts_insert_ms": 0.0,
        "commit_ms": 0.0,
        "structural_optimize_ms": 0.0,
        "lexical_optimize_ms": 0.0,
        "vocab_creation_ms": 0.0,
        "total_wall_ms": 0.0,
    }


def add_timing(target: dict[str, float], key: str, delta_ms: float) -> None:
    target[key] = float(target.get(key, 0.0)) + float(delta_ms)


def write_index_meta(meta_path: Path, payload: dict) -> None:
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, meta_path)


def insert_prepared_batch(
    con: sqlite3.Connection,
    prepared: list[tuple[int, str, str, str]],
    timings: dict[str, float],
) -> None:
    if not prepared:
        return

    t = time.perf_counter()
    con.executemany(
        "INSERT INTO docs(rowid, pid, signals) VALUES (?, ?, ?)",
        ((rowid, pid, signals) for rowid, pid, signals, _terms in prepared),
    )
    add_timing(timings, "docs_insert_ms", (time.perf_counter() - t) * 1000.0)

    t = time.perf_counter()
    con.executemany(
        "INSERT INTO structural_fts(rowid, signals) VALUES (?, ?)",
        ((rowid, signals) for rowid, _pid, signals, _terms in prepared),
    )
    add_timing(timings, "structural_fts_insert_ms", (time.perf_counter() - t) * 1000.0)

    t = time.perf_counter()
    con.executemany(
        "INSERT INTO lexical_fts(rowid, terms) VALUES (?, ?)",
        ((rowid, terms) for rowid, _pid, _signals, terms in prepared),
    )
    add_timing(timings, "lexical_fts_insert_ms", (time.perf_counter() - t) * 1000.0)

    t = time.perf_counter()
    con.commit()
    add_timing(timings, "commit_ms", (time.perf_counter() - t) * 1000.0)


def prepare_raw_batch(
    raw_batch: list[tuple[int, str, str]],
    indexer: MSMarcoIndexer,
    timings: dict[str, float],
) -> list[tuple[int, str, str, str]]:
    t = time.perf_counter()
    tokenized: list[tuple[int, str, list[str]]] = [
        (rowid, pid, indexer.tokenize(text)) for rowid, pid, text in raw_batch
    ]
    add_timing(timings, "tokenization_ms", (time.perf_counter() - t) * 1000.0)

    t = time.perf_counter()
    prepared = [
        (
            rowid,
            pid,
            " ".join(sorted(indexer.extract_signals_from_tokens(tokens))),
            " ".join(tokens),
        )
        for rowid, pid, tokens in tokenized
    ]
    add_timing(timings, "signal_extraction_ms", (time.perf_counter() - t) * 1000.0)
    return prepared


def finalize_unified_index(
    con: sqlite3.Connection,
    timings: dict[str, float],
    unsafe_fast_build: bool,
) -> None:
    print("Optimizing structural FTS5 index...")
    t = time.perf_counter()
    con.execute("INSERT INTO structural_fts(structural_fts) VALUES ('optimize')")
    con.commit()
    add_timing(timings, "structural_optimize_ms", (time.perf_counter() - t) * 1000.0)

    print("Optimizing lexical FTS5 index...")
    t = time.perf_counter()
    con.execute("INSERT INTO lexical_fts(lexical_fts) VALUES ('optimize')")
    con.commit()
    add_timing(timings, "lexical_optimize_ms", (time.perf_counter() - t) * 1000.0)

    print("Creating FTS5 vocabulary views...")
    t = time.perf_counter()
    for name in ("struct_vocab_row", "struct_vocab_instance", "lex_vocab_row", "lex_vocab_instance"):
        con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute("CREATE VIRTUAL TABLE struct_vocab_row USING fts5vocab(structural_fts, 'row')")
    con.execute("CREATE VIRTUAL TABLE struct_vocab_instance USING fts5vocab(structural_fts, 'instance')")
    con.execute("CREATE VIRTUAL TABLE lex_vocab_row USING fts5vocab(lexical_fts, 'row')")
    con.execute("CREATE VIRTUAL TABLE lex_vocab_instance USING fts5vocab(lexical_fts, 'instance')")
    con.commit()
    add_timing(timings, "vocab_creation_ms", (time.perf_counter() - t) * 1000.0)

    if not unsafe_fast_build:
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass


def build_index_from_collection(
    con: sqlite3.Connection,
    db_path: Path,
    meta_path: Path,
    signature: dict,
    collection_path: Path,
    max_docs: int | None,
    batch_size: int,
    log_interval: int,
    unsafe_fast_build: bool,
    existing_meta: dict | None,
) -> dict:
    indexer = MSMarcoIndexer(max_signals_per_document=64, min_token_len=2)
    timings = dict((existing_meta or {}).get("offline_timings_ms") or fresh_timing_dict())
    for key, value in fresh_timing_dict().items():
        timings.setdefault(key, value)

    processed = int(con.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
    if processed:
        print(f"Resuming unified index at document {processed + 1:,}.")

    session_start = time.perf_counter()
    raw_batch: list[tuple[int, str, str]] = []
    last_report = processed

    def flush_batch() -> None:
        nonlocal raw_batch, processed, last_report
        if not raw_batch:
            return
        prepared = prepare_raw_batch(raw_batch, indexer, timings)
        insert_prepared_batch(con, prepared, timings)
        processed += len(raw_batch)
        raw_batch.clear()
        del prepared
        gc.collect()

        meta = {
            **signature,
            "complete": False,
            "build_source": "collection.tsv",
            "collection_path": str(collection_path.resolve()),
            "document_count": processed,
            "offline_timings_ms": timings,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_index_meta(meta_path, meta)

        if log_interval and (processed - last_report >= log_interval):
            elapsed = time.perf_counter() - session_start
            session_docs = max(1, processed - int((existing_meta or {}).get("document_count", 0)))
            rate = session_docs / elapsed if elapsed else 0.0
            print(
                f"[UNIFIED INDEX] {processed:,} docs | {rate:,.0f} docs/s | "
                f"DB={format_gib(db_size_bytes(db_path))}"
            )
            last_report = processed

    for doc in iter_collection(collection_path, max_docs, start_after=processed):
        rowid = int(doc["ordinal"])
        raw_batch.append((rowid, str(doc["id"]), str(doc["text"])))
        if len(raw_batch) >= batch_size:
            flush_batch()
    flush_batch()

    finalize_unified_index(con, timings, unsafe_fast_build)
    add_timing(timings, "total_wall_ms", (time.perf_counter() - session_start) * 1000.0)

    n_docs = int(con.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
    n_signals = int(con.execute("SELECT COUNT(*) FROM struct_vocab_row").fetchone()[0])
    meta = {
        **signature,
        "complete": True,
        "build_source": "collection.tsv",
        "collection_path": str(collection_path.resolve()),
        "document_count": n_docs,
        "unique_structural_signals": n_signals,
        "offline_timings_ms": timings,
        "idf_storage": "not materialized; document frequency read on demand from struct_vocab_row",
        "created_or_completed_utc": datetime.now(timezone.utc).isoformat(),
        "database_size_bytes": db_size_bytes(db_path),
    }
    write_index_meta(meta_path, meta)
    return meta


def build_index_from_legacy_pickle(
    con: sqlite3.Connection,
    db_path: Path,
    meta_path: Path,
    signature: dict,
    legacy_features_path: Path,
    max_docs: int | None,
    batch_size: int,
    log_interval: int,
    unsafe_fast_build: bool,
) -> dict:
    """
    One-time migration path for the old cache_features_full.pkl.

    pickle cannot be streamed: the complete legacy dictionary is deserialized in
    RAM. Use this only on a machine with enough memory. The 1.2-GB legacy IDF
    pickle is not required because the unified FTS index derives exact document
    frequencies directly from structural postings.
    """
    print(f"Loading legacy feature pickle into RAM: {legacy_features_path}")
    print("WARNING: this requires enough RAM for the complete deserialized object graph.")
    load_start = time.perf_counter()
    with legacy_features_path.open("rb") as f:
        features = pickle.load(f)
    legacy_load_ms = (time.perf_counter() - load_start) * 1000.0
    if not hasattr(features, "items"):
        raise TypeError("Legacy feature pickle must contain a mapping keyed by document ID.")

    indexer = MSMarcoIndexer(max_signals_per_document=64, min_token_len=2)
    timings = fresh_timing_dict()
    timings["legacy_pickle_load_ms"] = legacy_load_ms
    session_start = time.perf_counter()
    prepared: list[tuple[int, str, str, str]] = []
    processed = 0

    for rowid, (pid_key, feature) in enumerate(features.items(), start=1):
        if max_docs is not None and processed >= max_docs:
            break
        pid = str(getattr(feature, "agent_name", pid_key))
        source_text = str(getattr(feature, "source_text", ""))
        graph_signals = set(getattr(feature, "graph_signals", set()))

        t = time.perf_counter()
        tokens = indexer.tokenize(source_text)
        add_timing(timings, "tokenization_ms", (time.perf_counter() - t) * 1000.0)

        prepared.append((rowid, pid, " ".join(sorted(graph_signals)), " ".join(tokens)))
        processed += 1
        if len(prepared) >= batch_size:
            insert_prepared_batch(con, prepared, timings)
            prepared.clear()
            gc.collect()
            if log_interval and processed % log_interval < batch_size:
                print(
                    f"[LEGACY MIGRATION] {processed:,} docs | "
                    f"DB={format_gib(db_size_bytes(db_path))}"
                )
                meta = {
                    **signature,
                    "complete": False,
                    "build_source": "legacy_feature_pickle",
                    "legacy_features_path": str(legacy_features_path.resolve()),
                    "document_count": processed,
                    "offline_timings_ms": timings,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
                write_index_meta(meta_path, meta)

    if prepared:
        insert_prepared_batch(con, prepared, timings)
        prepared.clear()

    del features
    gc.collect()

    finalize_unified_index(con, timings, unsafe_fast_build)
    add_timing(timings, "total_wall_ms", (time.perf_counter() - session_start) * 1000.0)
    n_docs = int(con.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
    n_signals = int(con.execute("SELECT COUNT(*) FROM struct_vocab_row").fetchone()[0])
    meta = {
        **signature,
        "complete": True,
        "build_source": "legacy_feature_pickle",
        "legacy_features_path": str(legacy_features_path.resolve()),
        "document_count": n_docs,
        "unique_structural_signals": n_signals,
        "offline_timings_ms": timings,
        "idf_storage": "not materialized; document frequency read on demand from struct_vocab_row",
        "created_or_completed_utc": datetime.now(timezone.utc).isoformat(),
        "database_size_bytes": db_size_bytes(db_path),
    }
    write_index_meta(meta_path, meta)
    return meta


def build_or_open_unified_index(
    collection_path: Path,
    max_docs: int | None,
    cache_dir: Path,
    batch_size: int,
    log_interval: int,
    force_rebuild: bool,
    cache_mb: int,
    unsafe_fast_build: bool,
    migrate_legacy_features: Path | None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path, meta_path = unified_index_paths(cache_dir, max_docs)
    expected = unified_cache_signature(collection_path, max_docs)

    existing_meta: dict | None = None
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = None

    def signature_matches(meta: dict | None) -> bool:
        if not meta:
            return False
        return all(meta.get(k) == v for k, v in expected.items())

    if (
        not force_rebuild
        and db_path.exists()
        and signature_matches(existing_meta)
        and existing_meta.get("complete") is True
    ):
        print(f"Reusing validated unified RQ2 index: {db_path}")
        con = sqlite3.connect(str(db_path))
        configure_query_connection(con, cache_mb)
        return con, db_path, meta_path, existing_meta, True

    resumable = (
        not force_rebuild
        and db_path.exists()
        and signature_matches(existing_meta)
        and existing_meta.get("complete") is False
        and existing_meta.get("build_source") == "collection.tsv"
        and migrate_legacy_features is None
    )

    if not resumable:
        for p in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm"), meta_path):
            if p.exists():
                p.unlink()
        existing_meta = None

    con = sqlite3.connect(str(db_path))
    configure_build_connection(con, cache_mb, unsafe_fast_build)
    create_unified_schema(con)

    if migrate_legacy_features is not None and existing_meta is None:
        if not migrate_legacy_features.exists():
            raise FileNotFoundError(migrate_legacy_features)
        meta = build_index_from_legacy_pickle(
            con=con,
            db_path=db_path,
            meta_path=meta_path,
            signature=expected,
            legacy_features_path=migrate_legacy_features,
            max_docs=max_docs,
            batch_size=batch_size,
            log_interval=log_interval,
            unsafe_fast_build=unsafe_fast_build,
        )
    else:
        meta = build_index_from_collection(
            con=con,
            db_path=db_path,
            meta_path=meta_path,
            signature=expected,
            collection_path=collection_path,
            max_docs=max_docs,
            batch_size=batch_size,
            log_interval=log_interval,
            unsafe_fast_build=unsafe_fast_build,
            existing_meta=existing_meta,
        )

    con.close()
    con = sqlite3.connect(str(db_path))
    configure_query_connection(con, cache_mb)
    print(f"Unified RQ2 index ready: {format_gib(db_size_bytes(db_path))}")
    return con, db_path, meta_path, meta, False


def structural_offline_ms(index_meta: dict) -> float | None:
    t = index_meta.get("offline_timings_ms") or {}
    keys = (
        "tokenization_ms",
        "signal_extraction_ms",
        "docs_insert_ms",
        "structural_fts_insert_ms",
        "commit_ms",
        "structural_optimize_ms",
        "vocab_creation_ms",
    )
    vals = [float(t.get(k, 0.0)) for k in keys]
    return sum(vals) if vals else None


def lexical_offline_ms(index_meta: dict) -> float | None:
    t = index_meta.get("offline_timings_ms") or {}
    keys = (
        "tokenization_ms",
        "lexical_fts_insert_ms",
        "commit_ms",
        "lexical_optimize_ms",
        "vocab_creation_ms",
    )
    vals = [float(t.get(k, 0.0)) for k in keys]
    return sum(vals) if vals else None


# ---------------------------------------------------------------------------
# Structural retrieval over disk-backed index
# ---------------------------------------------------------------------------
def structural_query_weights(
    con: sqlite3.Connection,
    query_signals: set[str],
    n_docs: int,
    weighted: bool,
) -> dict[str, float]:
    if not query_signals:
        return {}
    if not weighted:
        return {signal: 1.0 for signal in query_signals}

    signals = sorted(query_signals)
    placeholders = ",".join("?" for _ in signals)
    rows = con.execute(
        f"SELECT term, doc FROM struct_vocab_row WHERE term IN ({placeholders})",
        signals,
    ).fetchall()
    return {
        str(term): math.log((n_docs + 1) / (int(df) + 1)) + 1.0
        for term, df in rows
    }


def rank_structural_isi(
    con: sqlite3.Connection,
    query: str,
    indexer: MSMarcoIndexer,
    n_docs: int,
    top_k: int,
    weighted: bool,
):
    query_signals = indexer.extract_query_signals(query)
    weights = structural_query_weights(con, query_signals, n_docs, weighted)
    if not weights:
        return [], 0, 0

    values_sql = ",".join("(?,?)" for _ in weights)
    params: list[object] = []
    for signal, weight in weights.items():
        params.extend([signal, float(weight)])
    params.append(top_k)

    sql = f"""
        WITH qterms(term, weight) AS (VALUES {values_sql}),
        scores AS (
            SELECT
                v.doc AS docid,
                SUM(q.weight) AS score,
                COUNT(*) AS matched_count
            FROM struct_vocab_instance AS v
            JOIN qterms AS q ON q.term = v.term
            GROUP BY v.doc
        ),
        ranked AS (
            SELECT
                d.pid AS pid,
                s.score AS score,
                s.matched_count AS matched_count,
                COUNT(*) OVER() AS candidate_count
            FROM scores AS s
            JOIN docs AS d ON d.rowid = s.docid
        )
        SELECT pid, score, matched_count, candidate_count
        FROM ranked
        ORDER BY score DESC, matched_count DESC, pid DESC
        LIMIT ?
    """
    rows = con.execute(sql, params).fetchall()
    if not rows:
        return [], 0, 0
    candidate_count = int(rows[0][3])
    return [str(row[0]) for row in rows], candidate_count, candidate_count


def rank_structural_fullscan(
    con: sqlite3.Connection,
    query: str,
    indexer: MSMarcoIndexer,
    n_docs: int,
    top_k: int,
    fetch_size: int,
):
    query_signals = indexer.extract_query_signals(query)
    weights = structural_query_weights(con, query_signals, n_docs, weighted=True)
    if not weights:
        return [], 0, n_docs

    get_weight = weights.get
    heap: list[tuple[tuple[float, int, str], str]] = []
    candidate_count = 0
    cursor = con.execute("SELECT pid, signals FROM docs ORDER BY rowid")

    while True:
        rows = cursor.fetchmany(fetch_size)
        if not rows:
            break
        for pid, signals_text in rows:
            score = 0.0
            matched_count = 0
            # Every structural signal occurs at most once in signals_text.
            for signal in str(signals_text).split():
                weight = get_weight(signal)
                if weight is not None:
                    score += weight
                    matched_count += 1
            if matched_count == 0:
                continue
            candidate_count += 1
            pid_str = str(pid)
            key = (score, matched_count, pid_str)
            payload = (key, pid_str)
            if len(heap) < top_k:
                heapq.heappush(heap, payload)
            elif key > heap[0][0]:
                heapq.heapreplace(heap, payload)

    ranked = sorted(heap, key=lambda x: x[0], reverse=True)
    return [pid for _key, pid in ranked], candidate_count, n_docs


def evaluate_structural_isi_method(
    method: str,
    gold: Sequence[GoldQuery],
    con: sqlite3.Connection,
    indexer: MSMarcoIndexer,
    n_docs: int,
    top_k: int,
    log_interval: int,
    weighted: bool,
) -> list[dict]:
    details: list[dict] = []
    for i, item in enumerate(gold, start=1):
        start = time.perf_counter()
        ranked_ids, candidate_count, documents_scored = rank_structural_isi(
            con=con,
            query=item.query,
            indexer=indexer,
            n_docs=n_docs,
            top_k=top_k,
            weighted=weighted,
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


def evaluate_fullscan_method(
    gold: Sequence[GoldQuery],
    con: sqlite3.Connection,
    indexer: MSMarcoIndexer,
    n_docs: int,
    top_k: int,
    fetch_size: int,
    log_interval: int,
) -> list[dict]:
    details: list[dict] = []
    for i, item in enumerate(gold, start=1):
        start = time.perf_counter()
        ranked_ids, candidate_count, documents_scored = rank_structural_fullscan(
            con=con,
            query=item.query,
            indexer=indexer,
            n_docs=n_docs,
            top_k=top_k,
            fetch_size=fetch_size,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        rank = find_rank(ranked_ids, item.positive_ids)
        metrics = metric_values(rank)
        details.append({
            "method": METHOD_FULLSCAN,
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
        print(
            f"[{METHOD_FULLSCAN}] {i:,}/{len(gold):,} | "
            f"{latency_ms:,.2f} ms | candidates={candidate_count:,}"
        )
        if log_interval and i % log_interval == 0:
            avg = statistics.fmean(x["latency_ms"] for x in details)
            print(f"[{METHOD_FULLSCAN}] running avg={avg:,.2f} ms/query")
    return details


# ---------------------------------------------------------------------------
# Lexical baselines over the same unified index
# ---------------------------------------------------------------------------
def baseline_query_tokens(query: str, indexer: MSMarcoIndexer) -> list[str]:
    return indexer.tokenize(query)


def fts_quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def rank_bm25(
    con: sqlite3.Connection,
    query: str,
    indexer: MSMarcoIndexer,
    top_k: int,
):
    tokens = baseline_query_tokens(query, indexer)
    if not tokens:
        return [], None
    unique = list(dict.fromkeys(tokens))
    match_expr = " OR ".join(fts_quote(t) for t in unique)
    rows = con.execute(
        "SELECT d.pid, bm25(lexical_fts) AS score "
        "FROM lexical_fts JOIN docs AS d ON d.rowid = lexical_fts.rowid "
        "WHERE lexical_fts MATCH ? "
        "ORDER BY score ASC, d.pid ASC LIMIT ?",
        (match_expr, top_k),
    ).fetchall()
    return [str(r[0]) for r in rows], None


def rank_tfidf_dot(
    con: sqlite3.Connection,
    query: str,
    indexer: MSMarcoIndexer,
    top_k: int,
    n_docs: int,
):
    tokens = baseline_query_tokens(query, indexer)
    if not tokens:
        return [], 0
    qtf = Counter(tokens)
    terms = list(qtf.keys())
    placeholders = ",".join("?" for _ in terms)
    df_rows = con.execute(
        f"SELECT term, doc FROM lex_vocab_row WHERE term IN ({placeholders})",
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
        FROM lex_vocab_instance AS v
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
    indexer: MSMarcoIndexer,
    n_docs: int,
    top_k: int,
    log_interval: int,
) -> list[dict]:
    details: list[dict] = []
    for i, item in enumerate(gold, start=1):
        start = time.perf_counter()
        if method == METHOD_BM25:
            ranked_ids, candidate_count = rank_bm25(con, item.query, indexer, top_k)
        elif method == METHOD_TFIDF:
            ranked_ids, candidate_count = rank_tfidf_dot(con, item.query, indexer, top_k, n_docs)
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
# Output / checkpoints
# ---------------------------------------------------------------------------
def method_safe_name(method: str) -> str:
    return (
        method.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def write_method_checkpoint(run_dir: Path, method: str, details: list[dict], summary: EvalSummary) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    safe = method_safe_name(method)
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
) -> None:
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
# CLI / main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RQ2 disk-backed ablation benchmark")
    p.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    p.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--run-name", default=None, help="Output subdirectory; default is a UTC timestamp.")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--latency-sample",
        type=int,
        default=DEFAULT_LATENCY_SAMPLE,
        help="Deterministic query sample used by the exhaustive full-scan ablation.",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=100_000,
        help="Smoke-test corpus cap. Use --official for the full collection.",
    )
    p.add_argument(
        "--max-queries",
        type=int,
        default=200,
        help="Smoke-test query cap. Use --official for all dev-small eligible queries.",
    )
    p.add_argument(
        "--official",
        action="store_true",
        help=f"Use up to {OFFICIAL_MAX_DOCS:,} docs and {OFFICIAL_MAX_QUERIES:,} queries.",
    )
    p.add_argument(
        "--methods",
        type=parse_methods,
        default=parse_methods("tfidf,bm25,overlap,fullscan,ssr"),
        help="Comma-separated: tfidf,bm25,overlap,fullscan,ssr",
    )
    p.add_argument(
        "--full-scan-all",
        action="store_true",
        help="Exhaustively scan the corpus separately for every eligible query. Extremely expensive.",
    )
    p.add_argument(
        "--index-only",
        action="store_true",
        help="Build/reuse the unified disk index and exit before loading/evaluating queries.",
    )
    p.add_argument("--index-batch-size", type=int, default=DEFAULT_INDEX_BATCH_SIZE)
    p.add_argument("--fullscan-fetch-size", type=int, default=DEFAULT_FULLSCAN_FETCH_SIZE)
    p.add_argument("--sqlite-cache-mb", type=int, default=DEFAULT_SQLITE_CACHE_MB)
    p.add_argument("--force-index-rebuild", action="store_true")
    # Backward-compatible alias used by older commands.
    p.add_argument("--force-lexical-rebuild", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--unsafe-fast-build",
        action="store_true",
        help="Use journal_mode=OFF/synchronous=OFF while building. Faster, but interrupted builds may be unusable.",
    )
    p.add_argument(
        "--migrate-legacy-features",
        type=Path,
        default=None,
        help=(
            "Optional path to old cache_features_full.pkl. Builds the new unified disk index from the "
            "legacy feature mapping instead of re-extracting structural signals from collection.tsv. "
            "The pickle is loaded completely into RAM."
        ),
    )
    p.add_argument("--index-log-interval", type=int, default=100_000)
    p.add_argument("--query-log-interval", type=int, default=100)
    return p.parse_args()


def main() -> None:
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

    if not args.collection.exists():
        raise FileNotFoundError(args.collection)

    print("===== RQ2 DISK-BACKED ABLATION BENCHMARK =====")
    print(f"Protocol version: {PROTOCOL_VERSION}")
    print(f"Collection: {args.collection}")
    print(f"Max docs: {max_docs if max_docs is not None else 'ALL'}")
    print(f"Max queries: {max_queries if max_queries is not None else 'ALL'}")
    print(f"Top K: {args.top_k}")
    print(f"Methods: {methods}")
    print(f"Index batch size: {args.index_batch_size:,}")
    print(f"Run directory: {run_dir}")

    force_rebuild = args.force_index_rebuild or args.force_lexical_rebuild
    con, db_path, meta_path, index_meta, index_reused = build_or_open_unified_index(
        collection_path=args.collection,
        max_docs=max_docs,
        cache_dir=args.cache_dir,
        batch_size=args.index_batch_size,
        log_interval=args.index_log_interval,
        force_rebuild=force_rebuild,
        cache_mb=args.sqlite_cache_mb,
        unsafe_fast_build=args.unsafe_fast_build,
        migrate_legacy_features=args.migrate_legacy_features,
    )

    n_docs = int(index_meta["document_count"])
    print(f"Indexed documents: {n_docs:,}")
    print(f"Unique structural signals: {int(index_meta.get('unique_structural_signals', 0)):,}")
    print(f"Index reused in this run: {index_reused}")
    print(f"Unified index size: {format_gib(db_size_bytes(db_path))}")

    if args.index_only:
        print("Index-only mode complete. No queries or retrieval methods were executed.")
        con.close()
        return

    for path in (args.queries, args.qrels):
        if not path.exists():
            con.close()
            raise FileNotFoundError(path)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    print(f"Loaded queries file: {len(queries):,}")
    print(f"Loaded qrel query IDs: {len(qrels):,}")

    if n_docs >= OFFICIAL_MAX_DOCS and (max_docs is None or max_docs >= OFFICIAL_MAX_DOCS):
        doc_exists = None
    else:
        exists_cur = con.cursor()

        def doc_exists(pid: str) -> bool:
            return exists_cur.execute("SELECT 1 FROM docs WHERE pid=? LIMIT 1", (str(pid),)).fetchone() is not None

    gold = build_gold(queries, qrels, max_queries=max_queries, doc_exists=doc_exists)
    print(f"Eligible queries: {len(gold):,}")
    if not gold:
        raise RuntimeError("No eligible queries remain after qrel/document filtering.")

    rng = random.Random(args.seed)
    sample_size = min(args.latency_sample, len(gold))
    latency_indices = sorted(rng.sample(range(len(gold)), sample_size))
    latency_gold = [gold[i] for i in latency_indices]
    latency_query_ids = {item.query_id for item in latency_gold}
    print(f"Common deterministic full-scan sample: {len(latency_gold):,} queries (seed={args.seed})")

    indexer = MSMarcoIndexer(max_signals_per_document=64, min_token_len=2)
    structural_build_ms = structural_offline_ms(index_meta)
    lexical_build_ms = lexical_offline_ms(index_meta)
    cache_note = "cache reused" if index_reused else "index built in this run"

    summaries: list[EvalSummary] = []
    details_by_method: dict[str, list[dict]] = {}
    summary_by_method: dict[str, EvalSummary] = {}

    if METHOD_TFIDF in methods:
        print(f"\nRunning {METHOD_TFIDF} on {len(gold):,} queries...")
        d = evaluate_lexical_method(
            METHOD_TFIDF, gold, con, indexer, n_docs, args.top_k, args.query_log_interval
        )
        details_by_method[METHOD_TFIDF] = d
        s = summarize_method(
            METHOD_TFIDF,
            d,
            lexical_build_ms,
            f"Unigram TF-IDF dot product; shared one-pass disk index ({cache_note}). Offline time is original construction time, not cache-open time.",
            latency_query_ids=latency_query_ids,
        )
        summaries.append(s)
        summary_by_method[METHOD_TFIDF] = s
        write_method_checkpoint(run_dir, METHOD_TFIDF, d, s)

    if METHOD_BM25 in methods:
        print(f"\nRunning {METHOD_BM25} on {len(gold):,} queries...")
        d = evaluate_lexical_method(
            METHOD_BM25, gold, con, indexer, n_docs, args.top_k, args.query_log_interval
        )
        details_by_method[METHOD_BM25] = d
        s = summarize_method(
            METHOD_BM25,
            d,
            lexical_build_ms,
            f"SQLite FTS5 BM25 over normalized unigrams; shared one-pass disk index ({cache_note}).",
            latency_query_ids=latency_query_ids,
        )
        summaries.append(s)
        summary_by_method[METHOD_BM25] = s
        write_method_checkpoint(run_dir, METHOD_BM25, d, s)

    if METHOD_OVERLAP in methods:
        print(f"\nRunning {METHOD_OVERLAP} on {len(gold):,} queries...")
        d = evaluate_structural_isi_method(
            METHOD_OVERLAP,
            gold,
            con,
            indexer,
            n_docs,
            args.top_k,
            args.query_log_interval,
            weighted=False,
        )
        details_by_method[METHOD_OVERLAP] = d
        s = summarize_method(
            METHOD_OVERLAP,
            d,
            structural_build_ms,
            f"Unweighted structural overlap; disk-backed ISI restricts scoring to matching documents ({cache_note}).",
            latency_query_ids=latency_query_ids,
        )
        summaries.append(s)
        summary_by_method[METHOD_OVERLAP] = s
        write_method_checkpoint(run_dir, METHOD_OVERLAP, d, s)

    if METHOD_SSR in methods:
        print(f"\nRunning {METHOD_SSR} on {len(gold):,} queries...")
        d = evaluate_structural_isi_method(
            METHOD_SSR,
            gold,
            con,
            indexer,
            n_docs,
            args.top_k,
            args.query_log_interval,
            weighted=True,
        )
        details_by_method[METHOD_SSR] = d
        s = summarize_method(
            METHOD_SSR,
            d,
            structural_build_ms,
            f"Complete SSR structural-IDF ranking over the disk-backed ISI ({cache_note}).",
            latency_query_ids=latency_query_ids,
        )
        summaries.append(s)
        summary_by_method[METHOD_SSR] = s
        write_method_checkpoint(run_dir, METHOD_SSR, d, s)

    if METHOD_FULLSCAN in methods:
        if METHOD_SSR not in details_by_method:
            raise RuntimeError(
                "The full-scan ablation requires 'ssr' in --methods so exact ranking equivalence can be verified."
            )

        if args.full_scan_all:
            fullscan_gold = list(gold)
            direct_effectiveness = True
            note = (
                "Exhaustive structural-IDF scoring over every indexed document for every eligible query. "
                "This is intentionally very expensive."
            )
        else:
            fullscan_gold = list(latency_gold)
            direct_effectiveness = False
            note = (
                f"Exhaustive structural-IDF scoring over all {n_docs:,} indexed documents for the deterministic "
                f"{len(fullscan_gold)}-query sample (seed={args.seed}). Effectiveness is inherited from SSR only "
                "after exact top-k equivalence is verified on this implementation check."
            )

        print(f"\nRunning {METHOD_FULLSCAN} on {len(fullscan_gold):,} queries...")
        d = evaluate_fullscan_method(
            fullscan_gold,
            con,
            indexer,
            n_docs,
            args.top_k,
            args.fullscan_fetch_size,
            args.query_log_interval,
        )
        details_by_method[METHOD_FULLSCAN] = d

        ssr_map = {
            x["query_id"]: x["top_document_ids"]
            for x in details_by_method[METHOD_SSR]
        }
        mismatches = [
            x["query_id"]
            for x in d
            if ssr_map.get(x["query_id"]) != x["top_document_ids"]
        ]
        if mismatches:
            raise RuntimeError(
                "Full-scan structural-IDF and SSR rankings differ for query IDs: "
                + ", ".join(mismatches[:10])
            )
        print("Verified: full-scan structural-IDF and SSR top-k rankings are identical on the deterministic sample.")

        s = summarize_method(
            METHOD_FULLSCAN,
            d,
            structural_build_ms,
            note,
            effectiveness_measured=direct_effectiveness,
            latency_query_ids=latency_query_ids,
        )
        if not direct_effectiveness:
            ssr_summary = summary_by_method[METHOD_SSR]
            copy_effectiveness(ssr_summary, s, len(gold))
        summaries.append(s)
        summary_by_method[METHOD_FULLSCAN] = s
        write_method_checkpoint(run_dir, METHOD_FULLSCAN, d, s)

    order = [METHOD_TFIDF, METHOD_BM25, METHOD_OVERLAP, METHOD_FULLSCAN, METHOD_SSR]
    summaries.sort(key=lambda x: order.index(x.method) if x.method in order else 999)

    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MS MARCO Passage Ranking Dev Small",
        "collection_path": str(args.collection.resolve()),
        "queries_path": str(args.queries.resolve()),
        "qrels_path": str(args.qrels.resolve()),
        "queries_sha256": sha256_file(args.queries),
        "qrels_sha256": sha256_file(args.qrels),
        "collection_fingerprint": quick_file_fingerprint(args.collection),
        "max_docs": max_docs,
        "max_queries": max_queries,
        "indexed_documents": n_docs,
        "eligible_queries": len(gold),
        "top_k": args.top_k,
        "seed": args.seed,
        "full_scan_latency_sample_requested": args.latency_sample,
        "full_scan_all": args.full_scan_all,
        "common_latency_query_count": len(latency_query_ids),
        "common_latency_query_ids": sorted(latency_query_ids),
        "methods": methods,
        "unified_index": {
            "db_path": str(db_path.resolve()),
            "meta_path": str(meta_path.resolve()),
            "reused": index_reused,
            "database_size_bytes": db_size_bytes(db_path),
            "schema_version": INDEX_SCHEMA_VERSION,
            "build_source": index_meta.get("build_source"),
            "offline_timings_ms": index_meta.get("offline_timings_ms"),
            "structural_offline_time_reported_ms": structural_build_ms,
            "lexical_offline_time_reported_ms": lexical_build_ms,
            "unique_structural_signals": index_meta.get("unique_structural_signals"),
            "idf_storage": index_meta.get("idf_storage"),
        },
        "structural_extraction": {
            "max_signals_per_document": 64,
            "min_token_len": 2,
            "signals": "normalized unigrams followed by consecutive bigrams; bounded to 64 unique signals",
            "extractor_version": EXTRACTOR_VERSION,
        },
        "idf_formula": "log((N+1)/(df+1)) + 1",
        "tfidf_formula": "sum_t qtf(t) * dtf(t,d) * idf(t); no cosine normalization",
        "bm25_implementation": "SQLite FTS5 bm25() over contentless lexical index",
        "isi_implementation": "SQLite FTS5 structural postings queried through fts5vocab(instance)",
        "fullscan_implementation": "sequential SELECT over docs table; every document visited for each sampled query",
        "latency_scope": "online query execution only; offline index construction reported separately",
        "fullscan_effectiveness_policy": (
            "When --full-scan-all is false, effectiveness is copied from SSR only after exact top-k "
            "equivalence is verified on the deterministic sample; the score function is identical and ISI "
            "only removes zero-overlap documents."
        ),
        "python_version": sys.version,
        "sqlite_version": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "script": str(THIS_FILE),
        "run_name": run_name,
    }

    write_outputs(run_dir, summaries, details_by_method, metadata)
    con.close()

    print("\n===== SUMMARY =====")
    print(json.dumps([asdict(x) for x in summaries], indent=2))


if __name__ == "__main__":
    main()
