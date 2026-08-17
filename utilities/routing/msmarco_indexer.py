from dataclasses import dataclass
from typing import Any
import re
from collections import Counter

from utilities.routing.agent_indexer import AgentFeature


@dataclass
class MSMarcoDocument:
    id: str
    text: str


class MSMarcoIndexer:
    """
    Converts MS MARCO passages into AgentFeature objects so the SSR structural
    representation can be reused by the retrieval benchmarks.

    The public tokenization/signal helpers are intentionally deterministic.
    RQ2 uses them to build structural and lexical disk indexes in a single pass
    over the collection without changing the signal semantics used elsewhere.
    """

    def __init__(
        self,
        max_signals_per_document: int = 64,
        min_token_len: int = 2,
    ):
        self.max_signals_per_document = max_signals_per_document
        self.min_token_len = min_token_len
        self.stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "have", "he", "her", "his", "how", "i",
            "in", "is", "it", "its", "of", "on", "or", "that", "the",
            "their", "this", "to", "was", "what", "when", "where",
            "which", "who", "why", "will", "with", "you", "your",
            "do", "does", "did", "can", "could", "would", "should",
            "about", "into", "than", "then", "there", "these", "those",
            "also", "were", "been", "being", "not", "but", "if",
        }

    def build_features(
        self,
        documents: list[dict[str, Any]],
    ) -> dict[str, AgentFeature]:
        features = {}
        for doc in documents:
            feature = self.index_document(doc)
            features[feature.agent_name] = feature
        return features

    def index_document(
        self,
        document: dict[str, Any],
    ) -> AgentFeature:
        document_id = str(document["id"])
        text = str(document["text"]).strip()
        graph_signals = self.extract_document_signals(text)
        specialty = self._infer_specialty(graph_signals)
        return AgentFeature(
            agent_name=document_id,
            agent_url=f"msmarco://{document_id}",
            description=text[:500],
            source_text=text.lower(),
            tree_keys={
                "domain": "general",
                "task_type": "answer",
                "modality": "text_to_text",
                "specialty": specialty,
            },
            graph_signals=graph_signals,
            constraints={
                "input_modes": ["text"],
                "output_modes": ["text"],
                "task_types": ["answer"],
                "domains": ["general"],
                "topics": [specialty] if specialty != "general" else [],
                "latency_tier": "medium",
                "cost_tier": "medium",
                "reliability_tier": "medium",
                "security_level": "low",
                "permissions": ["read"],
            },
        )

    def tokenize(self, text: str) -> list[str]:
        """Public deterministic tokenizer used by RQ2 lexical baselines."""
        return self._tokenize(text)

    def extract_signals_from_tokens(self, tokens: list[str]) -> set[str]:
        """
        Build the exact structural signal set from already-normalized tokens.

        This avoids tokenizing every MS MARCO passage twice when RQ2 builds the
        lexical and structural indexes in the same collection pass.
        """
        phrases = self._extract_phrases(tokens)
        counts = Counter(tokens)
        ranked_tokens = [
            token
            for token, _ in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        ranked_phrases = sorted(phrases)

        ordered_signals: list[str] = []
        seen: set[str] = set()
        for signal in ranked_tokens:
            if signal not in seen:
                ordered_signals.append(signal)
                seen.add(signal)
        for signal in ranked_phrases:
            if signal not in seen:
                ordered_signals.append(signal)
                seen.add(signal)

        return set(ordered_signals[: self.max_signals_per_document])

    def extract_document_signals(self, text: str) -> set[str]:
        """Extract structural signals from a passage using the SSR rules."""
        return self.extract_signals_from_tokens(self.tokenize(text))

    def extract_query_signals(self, query: str) -> set[str]:
        """Extract structural signals from a query without agent normalization."""
        return self.extract_document_signals(query)

    # Backward-compatible private helpers used by older benchmark scripts.
    def _extract_graph_signals(self, text: str) -> set[str]:
        return self.extract_document_signals(text)

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        raw_tokens = re.findall(r"[a-z0-9]+", text)

        tokens = []
        for token in raw_tokens:
            if len(token) < self.min_token_len:
                continue
            if token in self.stopwords:
                continue
            tokens.append(token)
        return tokens

    def _extract_phrases(self, tokens: list[str]) -> set[str]:
        phrases = set()
        for i in range(len(tokens) - 1):
            phrases.add(f"{tokens[i]}_{tokens[i + 1]}")
        return phrases

    def _infer_specialty(self, graph_signals: set[str]) -> str:
        if not graph_signals:
            return "general"
        bigrams = sorted(s for s in graph_signals if "_" in s)
        if bigrams:
            return bigrams[0]
        return sorted(graph_signals)[0]
