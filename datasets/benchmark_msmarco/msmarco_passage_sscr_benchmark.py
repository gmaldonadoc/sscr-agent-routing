# ============================================================
# MS MARCO Passage Ranking Benchmark
# SSCR + IDF-weighted Graph + Inverted Signal Index
# ============================================================

import os
import sys
import json
import time
import csv
import math
from collections import defaultdict


# ============================================================
# Project path setup
# ============================================================

root_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if root_dir not in sys.path:
    sys.path.append(root_dir)

from utilities.routing.msmarco_indexer import MSMarcoIndexer


# ============================================================
# Paths and experiment configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COLLECTION_PATH = os.path.join(BASE_DIR, "collection.tsv")
QUERIES_PATH = os.path.join(BASE_DIR, "queries.dev.small.tsv")
QRELS_PATH = os.path.join(BASE_DIR, "qrels.dev.small.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# Small test
MAX_DOCS = 100_000
MAX_QUERIES = 500

# Official benchmark
# MAX_DOCS = 8_841_823
# MAX_QUERIES = 6_980

TOP_K = 10
SAVE_DETAILS = True

INDEX_LOG_INTERVAL = 10_000
EVAL_LOG_INTERVAL = 100


# ============================================================
# Data loading
# ============================================================

def load_queries(path: str) -> dict[str, str]:
    queries = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) < 2:
                continue

            qid = str(row[0])
            query = row[1].strip()
            queries[qid] = query

    return queries


def load_qrels(path: str) -> dict[str, list[str]]:
    qrels = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) < 4:
                continue

            qid = str(row[0])
            pid = str(row[2])
            label = int(row[3])

            if label > 0:
                qrels[qid].append(pid)

    return dict(qrels)


def iter_collection(path: str, max_docs: int | None = None):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for idx, row in enumerate(reader, start=1):
            if max_docs is not None and idx > max_docs:
                break

            if len(row) < 2:
                continue

            yield {
                "id": str(row[0]),
                "text": row[1],
            }


def count_collection_rows(path: str, max_docs: int | None = None) -> int:
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for count, _ in enumerate(f, start=1):
            if max_docs is not None and count >= max_docs:
                break

    return count


# ============================================================
# Gold construction
# ============================================================

def build_gold(
    queries: dict[str, str],
    qrels: dict[str, list[str]],
    valid_doc_ids: set[str],
    max_queries: int | None = None,
) -> list[dict]:
    gold = []

    for qid, positive_ids in qrels.items():
        if qid not in queries:
            continue

        valid_positive_ids = [
            pid for pid in positive_ids
            if pid in valid_doc_ids
        ]

        if not valid_positive_ids:
            continue

        gold.append({
            "query_id": qid,
            "query": queries[qid],
            "positive_ids": valid_positive_ids,
        })

        if max_queries is not None and len(gold) >= max_queries:
            break

    return gold


# ============================================================
# IDF computation
# ============================================================

def compute_signal_idf(features: dict) -> dict[str, float]:
    signal_df = defaultdict(int)
    total_docs = len(features)

    for feature in features.values():
        for signal in feature.graph_signals:
            signal_df[signal] += 1

    signal_idf = {}

    for signal, df in signal_df.items():
        signal_idf[signal] = math.log((total_docs + 1) / (df + 1)) + 1.0

    return signal_idf


# ============================================================
# Inverted Signal Index
# signal -> set(document_ids)
# ============================================================

def build_signal_index(features: dict) -> dict[str, set[str]]:
    signal_index = defaultdict(set)

    for doc_id, feature in features.items():
        for signal in feature.graph_signals:
            signal_index[signal].add(doc_id)

    return dict(signal_index)


def collect_candidate_ids(
    query_signals: set[str],
    signal_index: dict[str, set[str]],
) -> set[str]:
    candidate_ids = set()

    for signal in query_signals:
        candidate_ids.update(
            signal_index.get(signal, set())
        )

    return candidate_ids


# ============================================================
# Ranking
# ============================================================

def rank_documents(
    query: str,
    features: dict,
    indexer: MSMarcoIndexer,
    signal_idf: dict[str, float],
    signal_index: dict[str, set[str]],
    top_k: int = 10,
):
    query_signals = indexer.extract_query_signals(query)

    candidate_ids = collect_candidate_ids(
        query_signals=query_signals,
        signal_index=signal_index,
    )

    ranked = []

    for doc_id in candidate_ids:
        feature = features.get(doc_id)

        if feature is None:
            continue

        matched = query_signals.intersection(feature.graph_signals)

        if not matched:
            continue

        score = sum(
            signal_idf.get(signal, 1.0)
            for signal in matched
        )

        ranked.append({
            "document_id": doc_id,
            "score": round(score, 6),
            "matched_signals": sorted(matched),
        })

    ranked.sort(
        key=lambda x: (
            x["score"],
            len(x["matched_signals"]),
            x["document_id"],
        ),
        reverse=True,
    )

    return ranked[:top_k], len(candidate_ids)


# ============================================================
# Metrics
# ============================================================

def find_rank(ranked_docs: list[dict], positive_ids: list[str]):
    positives = set(str(pid) for pid in positive_ids)

    for idx, item in enumerate(ranked_docs, start=1):
        if str(item["document_id"]) in positives:
            return idx

    return None


def recall_at_k(rank, k: int) -> int:
    return 1 if rank is not None and rank <= k else 0


def mrr_at_10(rank) -> float:
    if rank is not None and rank <= 10:
        return 1.0 / rank

    return 0.0


# ============================================================
# Main evaluation
# ============================================================

def evaluate():
    # --------------------------------------------------------
    # Load official MS MARCO Dev Small files
    # --------------------------------------------------------

    print("Carregando queries oficiais MS MARCO Passage...")
    queries = load_queries(QUERIES_PATH)

    print("Carregando qrels oficiais MS MARCO Passage...")
    qrels = load_qrels(QRELS_PATH)

    print(f"Queries carregadas: {len(queries):,}")
    print(f"Queries com qrels: {len(qrels):,}")

    indexer = MSMarcoIndexer(
        max_signals_per_document=64,
        min_token_len=2,
    )

    # --------------------------------------------------------
    # Preprocessing / document feature indexing
    # --------------------------------------------------------

    print("Indexando collection.tsv...")
    start_index = time.perf_counter()

    features = {}

    total_docs = count_collection_rows(
        COLLECTION_PATH,
        max_docs=MAX_DOCS,
    )

    for idx, doc in enumerate(
        iter_collection(COLLECTION_PATH, max_docs=MAX_DOCS),
        start=1,
    ):
        feature = indexer.index_document(doc)
        features[feature.agent_name] = feature

        if idx % INDEX_LOG_INTERVAL == 0 or idx == total_docs:
            elapsed = time.perf_counter() - start_index
            docs_per_sec = idx / elapsed if elapsed > 0 else 0
            remaining = total_docs - idx
            eta_sec = remaining / docs_per_sec if docs_per_sec > 0 else 0

            print(
                f"[INDEX] {idx:,}/{total_docs:,} documentos "
                f"({idx / total_docs:.2%}) | "
                f"{docs_per_sec:,.1f} docs/s | "
                f"ETA: {eta_sec / 60:.1f} min"
            )

    indexing_time_ms = (time.perf_counter() - start_index) * 1000

    print(f"Features indexadas: {len(features):,}")
    print(f"Tempo de indexação: {indexing_time_ms:.2f} ms")

    # --------------------------------------------------------
    # IDF weighting
    # --------------------------------------------------------

    print("Calculando IDF dos sinais...")
    start_idf = time.perf_counter()

    signal_idf = compute_signal_idf(features)

    idf_time_ms = (time.perf_counter() - start_idf) * 1000

    print(f"Sinais com IDF: {len(signal_idf):,}")
    print(f"Tempo cálculo IDF: {idf_time_ms:.2f} ms")

    # --------------------------------------------------------
    # Inverted Signal Index
    # --------------------------------------------------------

    print("Construindo Inverted Signal Index...")
    start_signal_index = time.perf_counter()

    signal_index = build_signal_index(features)

    signal_index_time_ms = (
        time.perf_counter() - start_signal_index
    ) * 1000

    print(f"Sinais indexados: {len(signal_index):,}")
    print(f"Tempo índice invertido: {signal_index_time_ms:.2f} ms")

    # --------------------------------------------------------
    # Build eligible gold set
    # --------------------------------------------------------

    valid_doc_ids = set(features.keys())

    gold = build_gold(
        queries=queries,
        qrels=qrels,
        valid_doc_ids=valid_doc_ids,
        max_queries=MAX_QUERIES,
    )

    print(f"Queries elegíveis: {len(gold):,}")

    # --------------------------------------------------------
    # Query evaluation
    # --------------------------------------------------------

    metrics = defaultdict(float)
    detailed_results = []

    start_eval = time.perf_counter()

    for item in gold:
        query = item["query"]
        positive_ids = item["positive_ids"]

        start = time.perf_counter()

        ranked_docs, candidate_count = rank_documents(
            query=query,
            features=features,
            indexer=indexer,
            signal_idf=signal_idf,
            signal_index=signal_index,
            top_k=TOP_K,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        rank = find_rank(ranked_docs, positive_ids)

        metrics["total"] += 1
        metrics["recall_at_1"] += recall_at_k(rank, 1)
        metrics["recall_at_3"] += recall_at_k(rank, 3)
        metrics["recall_at_5"] += recall_at_k(rank, 5)
        metrics["recall_at_10"] += recall_at_k(rank, 10)
        metrics["mrr_at_10"] += mrr_at_10(rank)
        metrics["latency_ms"] += latency_ms
        metrics["avg_returned"] += len(ranked_docs)
        metrics["avg_candidates"] += candidate_count

        processed = int(metrics["total"])

        if processed % EVAL_LOG_INTERVAL == 0 or processed == len(gold):
            avg_latency = metrics["latency_ms"] / processed
            remaining_queries = len(gold) - processed
            eta_sec = (avg_latency / 1000.0) * remaining_queries
            elapsed_hours = (time.perf_counter() - start_eval) / 3600.0

            print(
                f"[EVAL] {processed:,}/{len(gold):,} queries "
                f"({processed / len(gold):.2%}) | "
                f"avg_latency={avg_latency:.2f} ms/query | "
                f"elapsed={elapsed_hours:.2f} h | "
                f"ETA={eta_sec / 60:.1f} min"
            )

        if SAVE_DETAILS:
            detailed_results.append({
                "query_id": item.get("query_id"),
                "query": query,
                "positive_ids": positive_ids,
                "rank": rank,
                "candidate_count": candidate_count,
                "hit_at_1": recall_at_k(rank, 1),
                "hit_at_3": recall_at_k(rank, 3),
                "hit_at_5": recall_at_k(rank, 5),
                "hit_at_10": recall_at_k(rank, 10),
                "mrr_at_10": mrr_at_10(rank),
                "latency_ms": round(latency_ms, 4),
                "returned_count": len(ranked_docs),
                "top_results": ranked_docs,
            })

    total = int(metrics["total"])

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {
        "experiment": "msmarco_passage_dev_small_sscr_idf_inverted_signal_index",
        "pipeline": "preprocessing + normalizer + idf-weighted graph + inverted signal index",
        "scoring": "idf_weighted_signal_overlap_with_inverted_index",
        "dataset": "MS MARCO Passage Ranking Dev Small",
        "collection_path": COLLECTION_PATH,
        "queries_path": QUERIES_PATH,
        "qrels_path": QRELS_PATH,
        "corpus_size": len(features),
        "queries_evaluated": total,
        "top_k": TOP_K,
        "indexing_time_ms": round(indexing_time_ms, 4),
        "idf_time_ms": round(idf_time_ms, 4),
        "signal_index_time_ms": round(signal_index_time_ms, 4),
        "num_idf_signals": len(signal_idf),
        "num_indexed_signals": len(signal_index),
        "avg_candidates_per_query": round(metrics["avg_candidates"] / total, 4) if total else 0.0,
        "recall_at_1": round(metrics["recall_at_1"] / total, 4) if total else 0.0,
        "recall_at_3": round(metrics["recall_at_3"] / total, 4) if total else 0.0,
        "recall_at_5": round(metrics["recall_at_5"] / total, 4) if total else 0.0,
        "recall_at_10": round(metrics["recall_at_10"] / total, 4) if total else 0.0,
        "mrr_at_10": round(metrics["mrr_at_10"] / total, 4) if total else 0.0,
        "avg_latency_ms": round(metrics["latency_ms"] / total, 4) if total else 0.0,
        "avg_returned_results": round(metrics["avg_returned"] / total, 4) if total else 0.0,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "msmarco_passage_sscr_idf_inverted_index_summary.json",
    )

    details_path = os.path.join(
        OUTPUT_DIR,
        "msmarco_passage_sscr_idf_inverted_index_details.json",
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if SAVE_DETAILS:
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print("\n===== MS MARCO PASSAGE SSCR IDF + INVERTED INDEX SUMMARY =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("\nArquivos gerados:")
    print(f"- {summary_path}")

    if SAVE_DETAILS:
        print(f"- {details_path}")


if __name__ == "__main__":
    evaluate()
