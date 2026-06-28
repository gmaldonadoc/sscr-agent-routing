import json
import random
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

DETAILS_PATH = RESULTS_DIR / "msmarco_passage_sscr_idf_inverted_index_details.json"
OUTPUT_PATH = BASE_DIR / "selected_200_queries.json"

RANDOM_STATE = 42

TARGET_EASY = 70      # rank = 1
TARGET_MEDIUM = 70    # rank 2-5
TARGET_HARD = 60      # rank 6-10


def difficulty_from_rank(rank: int) -> str:
    if rank == 1:
        return "easy"

    if 2 <= rank <= 5:
        return "medium"

    if 6 <= rank <= 10:
        return "hard"

    raise ValueError(f"Invalid rank: {rank}")


def main():
    print("Carregando detalhes do benchmark SSCR...")
    with open(DETAILS_PATH, "r", encoding="utf-8") as f:
        details = json.load(f)

    buckets = {
        "easy": [],
        "medium": [],
        "hard": [],
    }

    discarded = 0

    for item in details:
        rank = item.get("rank")

        if rank is None:
            discarded += 1
            continue

        if rank < 1 or rank > 10:
            discarded += 1
            continue

        difficulty = difficulty_from_rank(rank)

        buckets[difficulty].append({
            "query_id": item.get("query_id"),
            "query": item.get("query"),
            "positive_ids": item.get("positive_ids", []),
            "rank": rank,
            "difficulty": difficulty,
            "candidate_count": item.get("candidate_count"),
            "latency_ms": item.get("latency_ms"),
            "top_results": item.get("top_results", []),
        })

    rng = random.Random(RANDOM_STATE)

    targets = {
        "easy": TARGET_EASY,
        "medium": TARGET_MEDIUM,
        "hard": TARGET_HARD,
    }

    selected = []

    for difficulty, target in targets.items():
        bucket = buckets[difficulty]
        rng.shuffle(bucket)

        take = min(target, len(bucket))
        selected.extend(bucket[:take])

        print(
            f"{difficulty}: disponíveis={len(bucket):,}, "
            f"selecionadas={take}"
        )

    rng.shuffle(selected)

    counts = Counter(item["difficulty"] for item in selected)

    metadata = {
        "source_file": str(DETAILS_PATH),
        "selection_method": "stratified_by_sscr_rank",
        "random_state": RANDOM_STATE,
        "targets": targets,
        "available": {
            "easy": len(buckets["easy"]),
            "medium": len(buckets["medium"]),
            "hard": len(buckets["hard"]),
        },
        "selected_counts": dict(counts),
        "selected_total": len(selected),
        "discarded_rank_null_or_outside_top10": discarded,
        "total_input_queries": len(details),
    }

    output = {
        "metadata": metadata,
        "queries": selected,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nArquivo gerado:")
    print(OUTPUT_PATH)

    print("\nMetadata:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()