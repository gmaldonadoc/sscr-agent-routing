from utilities.routing.agent_indexer import AgentFeature
from utilities.routing.request_normalizer import RequestProfile
from utilities.routing.routing_memory import RoutingMemory


class SignalGraphRefiner:
    def __init__(self, agent_features: dict[str, AgentFeature]):
        self.agent_features = agent_features
        self.memory = RoutingMemory()

        self.signal_weights = {
            "utility": 3,
            "legal": 3,
            "finance": 3,
            "coding": 3,
            "greet": 3,
            "answer": 2,
            "extract": 3,
            "analyze": 3,
            "summarize": 3,
            "execute": 3,
            "time": 4,
            "greeting": 4,
            "contract": 3,
            "penalty_clause": 4,
            "terminal": 4,
            "file_system": 4,
            "text_input": 1,
            "document_input": 2,
            "text_output": 1,
            "table_output": 2,
            "json_output": 2,
        }

    def refine(self, request: RequestProfile, candidate_agents: list[str]) -> list[tuple[str, float]]:
        expanded_candidates = set(candidate_agents)

        # Expande candidatos quando há sinais fortes na query.
        # Exemplo: query de saudação + hora atual deve trazer GreetingAgent e TellTimeAgent.
        if self._should_expand_candidates(request):
            for agent_name, feature in self.agent_features.items():
                shared_signals = request.query_signals.intersection(feature.graph_signals)

                if self._has_strong_secondary_match(shared_signals):
                    expanded_candidates.add(agent_name)

        scored = []

        for agent_name in expanded_candidates:
            feature = self.agent_features.get(agent_name)
            if not feature:
                continue

            score = self._score(request.query_signals, feature.graph_signals, agent_name)

            if score > 0:
                scored.append((agent_name, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored
    
    def score_breakdown(
        self,
        query_signals: set[str],
        agent_signals: set[str],
        agent_name: str
    ) -> dict:
        signal_score = 0.0

        matched_signals = sorted(query_signals.intersection(agent_signals))

        for signal in matched_signals:
            signal_score += self.signal_weights.get(signal, 1)

        reputation = self.memory.get_agent_reputation(agent_name)
        reputation_boost = reputation * 2.0

        historical_success_boost = 0.0
        for signal in matched_signals:
            historical_success_boost += (
                self.memory.get_signal_success(signal, agent_name) * 1.0
            )

        co_occurrence_boost = self.memory.get_co_occurrence_boost(
            query_signals,
            agent_signals
        )

        total_score = (
            signal_score
            + reputation_boost
            + historical_success_boost
            + co_occurrence_boost
        )

        return {
            "matched_signals": matched_signals,
            "signal_score": round(signal_score, 3),
            "reputation_boost": round(reputation_boost, 3),
            "historical_success_boost": round(historical_success_boost, 3),
            "co_occurrence_boost": round(co_occurrence_boost, 3),
            "total_graph_score": round(total_score, 3),
        }

    def _should_expand_candidates(self, request: RequestProfile) -> bool:
        """
        Expande candidatos apenas quando a query parece composta/ambígua.
        Para queries simples, mantém o corte da árvore.
        """
        if len(request.task_types) > 1:
            return True

        if request.constraints.get("requires_external_action"):
            return True

        if len(request.topics) > 1:
            return True

        return False

    def _has_strong_secondary_match(self, shared_signals: set[str]) -> bool:
        strong_signals = {
            "time",
            "terminal",
            "file_system",
            "contract",
            "penalty_clause",
        }

        return bool(shared_signals.intersection(strong_signals))

    def _score(
        self,
        query_signals: set[str],
        agent_signals: set[str],
        agent_name: str | None = None
    ) -> float:
        if agent_name:
            return self.score_breakdown(
                query_signals=query_signals,
                agent_signals=agent_signals,
                agent_name=agent_name
            )["total_graph_score"]

        score = 0.0
        for signal in query_signals:
            if signal in agent_signals:
                score += self.signal_weights.get(signal, 1)

        return round(score, 3)