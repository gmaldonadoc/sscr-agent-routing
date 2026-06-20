from dataclasses import dataclass
from typing import Any
import re
import math
from collections import Counter

from utilities.routing.agent_indexer import AgentFeature


@dataclass
class MSMarcoDocument:
    id: str
    text: str


class MSMarcoIndexer:
    """
    Converte passagens MS MARCO em AgentFeature para reutilizar o SSCR.

    Experimento A:
    - não usa query_type
    - não usa constraints operacionais reais
    - usa texto da passagem para gerar graph_signals
    - usa tree_keys genéricas para manter compatibilidade com a árvore
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

        graph_signals = self._extract_graph_signals(text)
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

                # Neutro no Experimento A.
                # MS MARCO não possui esses metadados operacionais.
                "latency_tier": "medium",
                "cost_tier": "medium",
                "reliability_tier": "medium",
                "security_level": "low",
                "permissions": ["read"],
            },
        )

    def extract_query_signals(self, query: str) -> set[str]:
        """
        Use isto no benchmark MS MARCO para gerar sinais da query
        sem depender do RequestNormalizer de agentes.
        """
        return self._extract_graph_signals(query)

    def _extract_graph_signals(self, text: str) -> set[str]:
        tokens = self._tokenize(text)
        phrases = self._extract_phrases(tokens)

        counts = Counter(tokens)

        ranked_tokens = [
            token
            for token, _ in counts.most_common(self.max_signals_per_document)
        ]

        signals = set(ranked_tokens)
        signals.update(phrases)

        return set(list(signals)[: self.max_signals_per_document])

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

        # bigramas simples ajudam em casos como:
        # "reserve bank", "results based", "ronald reagan"
        for i in range(len(tokens) - 1):
            phrases.add(f"{tokens[i]}_{tokens[i + 1]}")

        return phrases

    def _infer_specialty(self, graph_signals: set[str]) -> str:
        """
        Como o MS MARCO não possui specialty real,
        usamos o sinal mais informativo como specialty artificial.
        Isso permite manter a árvore funcionando sem query_type.
        """
        if not graph_signals:
            return "general"

        # prefere bigramas porque são mais específicos que unigramas
        bigrams = sorted(s for s in graph_signals if "_" in s)
        if bigrams:
            return bigrams[0]

        return sorted(graph_signals)[0]