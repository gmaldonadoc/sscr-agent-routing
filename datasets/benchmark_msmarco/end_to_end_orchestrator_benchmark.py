from dotenv import load_dotenv
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

print("API Key carregada?", os.getenv("OPENAI_API_KEY") is not None)

import os
import sys
import csv
import json
import time
import math
import random
import statistics
from pathlib import Path
from collections import defaultdict

from openai import OpenAI



root_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if root_dir not in sys.path:
    sys.path.append(root_dir)

from utilities.routing.msmarco_indexer import MSMarcoIndexer


# ============================================================
# Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

COLLECTION_PATH = BASE_DIR / "collection.tsv"
QUERIES_PATH = BASE_DIR / "queries.dev.small.tsv"
QRELS_PATH = BASE_DIR / "qrels.dev.small.tsv"
OUTPUT_DIR = BASE_DIR / "results"

#N_VALUES = [100]
N_VALUES = [100, 500, 1000, 5000]
NUM_QUERIES = 6980


SEED = 42

MODEL = "gpt-4o-mini"

# Ajuste conforme o modelo/preço usado.
INPUT_PRICE_PER_1M = 0.15
OUTPUT_PRICE_PER_1M = 0.60

MAX_CONTEXT_TOKENS = 120_000


client = OpenAI()

# ============================================================
# Utils
# ============================================================

def estimate_tokens(text: str) -> int:
    # Aproximação simples: ~4 chars/token.
    return max(1, len(text) // 4)


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_1M
        +
        output_tokens / 1_000_000 * OUTPUT_PRICE_PER_1M
    )


def load_queries(path: Path) -> dict[str, str]:
    queries = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) >= 2:
                queries[str(row[0])] = row[1].strip()

    return queries


def load_qrels(path: Path) -> dict[str, list[str]]:
    qrels = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) >= 4 and int(row[3]) > 0:
                qrels[str(row[0])].append(str(row[2]))

    return dict(qrels)


def load_required_documents(
    collection_path: Path,
    required_pids: set[str],
    max_negative_docs: int,
) -> dict[str, str]:
    docs = {}
    negative_count = 0

    with open(collection_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) < 2:
                continue

            pid = str(row[0])
            text = row[1].strip()

            if pid in required_pids:
                docs[pid] = text

            elif negative_count < max_negative_docs:
                docs[pid] = text
                negative_count += 1

            if required_pids.issubset(docs.keys()) and negative_count >= max_negative_docs:
                break

    return docs


# ============================================================
# SSCR + IDF + Inverted Signal Index
# ============================================================

def compute_signal_idf(features: dict) -> dict[str, float]:
    signal_df = defaultdict(int)
    total_docs = len(features)

    for feature in features.values():
        for signal in feature.graph_signals:
            signal_df[signal] += 1

    return {
        signal: math.log((total_docs + 1) / (df + 1)) + 1.0
        for signal, df in signal_df.items()
    }


def build_signal_index(features: dict) -> dict[str, set[str]]:
    signal_index = defaultdict(set)

    for doc_id, feature in features.items():
        for signal in feature.graph_signals:
            signal_index[signal].add(doc_id)

    return dict(signal_index)


def sscr_filter_candidates(
    query: str,
    candidates: list[dict],
    indexer: MSMarcoIndexer,
) -> tuple[list[dict], float]:
    start = time.perf_counter()

    features = {}

    for doc in candidates:
        feature = indexer.index_document({
            "id": doc["pid"],
            "text": doc["text"],
        })
        features[feature.agent_name] = feature

    signal_idf = compute_signal_idf(features)
    signal_index = build_signal_index(features)

    query_signals = indexer.extract_query_signals(query)

    candidate_ids = set()

    for signal in query_signals:
        candidate_ids.update(signal_index.get(signal, set()))

    ranked = []

    for doc_id in candidate_ids:
        feature = features[doc_id]
        matched = query_signals.intersection(feature.graph_signals)

        if not matched:
            continue

        score = sum(signal_idf.get(signal, 1.0) for signal in matched)

        ranked.append({
            "pid": doc_id,
            "text": next(d["text"] for d in candidates if d["pid"] == doc_id),
            "sscr_score": score,
            "matched_signals": sorted(matched),
        })

    ranked.sort(
        key=lambda x: (
            x["sscr_score"],
            len(x["matched_signals"]),
            x["pid"],
        ),
        reverse=True,
    )

    latency_s = time.perf_counter() - start

    return ranked, latency_s


# ============================================================
# Prompt + LLM Orchestrator
# ============================================================

def build_prompt(query: str, candidates: list[dict]) -> str:
    docs_text = []

    for idx, doc in enumerate(candidates, start=1):
        docs_text.append(
            f"Document {idx}\n"
            f"pid: {doc['pid']}\n"
            f"text: {doc['text']}\n"
        )

    return (
        "You are a retrieval orchestrator.\n"
        "Given a user query and candidate passages, select the single passage "
        "that best answers the query.\n\n"
        "Return only valid JSON in this format:\n"
        "{\"selected_pid\": \"...\"}\n\n"
        f"Query:\n{query}\n\n"
        "Candidate passages:\n"
        + "\n".join(docs_text)
    )


def call_orchestrator(prompt: str) -> tuple[str | None, int, int, float]:
    input_tokens = estimate_tokens(prompt)

    if input_tokens > MAX_CONTEXT_TOKENS:
        return None, input_tokens, 0, 0.0

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    latency_s = time.perf_counter() - start

    content = response.choices[0].message.content
    output_tokens = estimate_tokens(content)

    try:
        parsed = json.loads(content)
        selected_pid = str(parsed.get("selected_pid"))
    except Exception:
        selected_pid = None

    return selected_pid, input_tokens, output_tokens, latency_s


# ============================================================
# Candidate construction
# ============================================================

def build_candidate_pool(
    qid: str,
    qrels: dict[str, list[str]],
    docs: dict[str, str],
    n: int,
    seed: int,
) -> list[dict]:
    positives = [
        pid for pid in qrels[qid]
        if pid in docs
    ]

    positive_set = set(positives)

    negatives = [
        pid for pid in docs.keys()
        if pid not in positive_set
    ]

    rng = random.Random(f"{seed}-{qid}-{n}")
    rng.shuffle(negatives)

    selected_pids = positives + negatives[: max(0, n - len(positives))]
    rng.shuffle(selected_pids)

    return [
        {
            "pid": pid,
            "text": docs[pid],
        }
        for pid in selected_pids
        if pid in docs
    ]


# ============================================================
# Evaluation
# ============================================================

def summarize_runs(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)

    for row in rows:
        grouped[(row["N_agents"], row["metodo"])].append(row)

    summary = []

    for (n, method), items in grouped.items():
        latencies = [x["latencia_s"] for x in items]
        accuracies = [x["accuracy"] for x in items]
        tokens = [x["tokens_prompt"] for x in items]
        costs = [x["custo_usd"] for x in items]
        ks = [x["K"] for x in items]
        valid_rates = [x["valid_selection"] for x in items] 
        

        summary.append({
            "N_agents": n,
            "Metodo": method,
            "K_mean": round(statistics.mean(ks), 4),
            "K_min": min(ks),
            "K_max": max(ks),
            "K_std": statistics.stdev(ks),
            "Tokens_prompt_mean": round(statistics.mean(tokens), 2),
            "Custo_usd_mean": round(statistics.mean(costs), 8),
            "Latencia_mean_s": round(statistics.mean(latencies), 4),
            "Latencia_std_s": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
            "Accuracy": round(statistics.mean(accuracies), 4),
            "Valid_selection_rate": round(statistics.mean(valid_rates), 4),
            "num_queries": len(items),
        })

    summary.sort(key=lambda x: (x["N_agents"], x["Metodo"]))

    return summary


def evaluate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando queries e qrels...")
    queries = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)

    selected_qids = list(qrels.keys())[:NUM_QUERIES]

    required_pids = set()

    for qid in selected_qids:
        required_pids.update(qrels[qid])

   

    print("Carregando documentos necessários da collection...")
    docs = load_required_documents(
        collection_path=COLLECTION_PATH,
        required_pids=required_pids,
        max_negative_docs=50_000,
    )

    print(f"Documentos carregados: {len(docs):,}")
    print(f"Queries avaliadas: {len(selected_qids):,}")

    indexer = MSMarcoIndexer(
        max_signals_per_document=64,
        min_token_len=2,
    )

    all_rows = []

    for n in N_VALUES:
        print(f"\n===== N = {n} =====")

        for qid in selected_qids:
            query = queries[qid]
            positive_ids = set(qrels[qid])

            candidates = build_candidate_pool(
                qid=qid,
                qrels=qrels,
                docs=docs,
                n=n,
                seed=SEED,
            )

            # -------------------------------
            # Baseline tradicional
            # -------------------------------

            prompt = build_prompt(query, candidates)

            selected_pid, in_tok, out_tok, llm_latency = call_orchestrator(prompt)
            candidate_ids = {doc["pid"] for doc in candidates}
            valid_selection = 1 if selected_pid in candidate_ids else 0
            all_rows.append({
                "query_id": qid,
                "N_agents": n,
                "metodo": "tradicional",
                "K": len(candidates),
                "tokens_prompt": in_tok,
                "tokens_output": out_tok,
                "custo_usd": compute_cost(in_tok, out_tok),
                "latencia_s": llm_latency,
                "accuracy": 1 if selected_pid in positive_ids else 0,
                "valid_selection": valid_selection,
                "selected_pid": selected_pid,
                "positive_ids": list(positive_ids),
            })
            
            # -------------------------------
            # SSCR + IDF + Inverted Index
            # -------------------------------

            sscr_candidates, sscr_latency = sscr_filter_candidates(
                query=query,
                candidates=candidates,
                indexer=indexer,
               
            )

            if not sscr_candidates:
                sscr_candidates = candidates

            prompt = build_prompt(query, sscr_candidates)

            selected_pid, in_tok, out_tok, llm_latency = call_orchestrator(prompt)

            sscr_candidate_ids = {doc["pid"] for doc in sscr_candidates}
            valid_selection = 1 if selected_pid in sscr_candidate_ids else 0

            all_rows.append({
                "query_id": qid,
                "N_agents": n,
                "metodo": "SSCR_IDF_InvertedIndex",
                "K": len(sscr_candidates),
                "tokens_prompt": in_tok,
                "tokens_output": out_tok,
                "custo_usd": compute_cost(in_tok, out_tok),
                "latencia_s": sscr_latency + llm_latency,
                "sscr_latency_s": sscr_latency,
                "llm_latency_s": llm_latency,
                "accuracy": 1 if selected_pid in positive_ids else 0,
                "valid_selection": valid_selection,
                "selected_pid": selected_pid,
                "positive_ids": list(positive_ids),
            })

    summary = summarize_runs(all_rows)

    details_path = OUTPUT_DIR / "end_to_end_orchestrator_details.json"
    summary_path = OUTPUT_DIR / "end_to_end_orchestrator_summary.json"
    csv_path = OUTPUT_DIR / "end_to_end_orchestrator_summary.csv"

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nArquivos gerados:")
    print(summary_path)
    print(details_path)
    print(csv_path)


if __name__ == "__main__":
    evaluate()