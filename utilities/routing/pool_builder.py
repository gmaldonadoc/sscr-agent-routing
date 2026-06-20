from utilities.routing.agent_indexer import AgentFeature
from utilities.routing.request_normalizer import RequestProfile


class CompactPoolBuilder:
    def __init__(self, agent_features: dict[str, AgentFeature], default_size: int = 5):
        self.agent_features = agent_features
        self.default_size = default_size

    def build(self, request: RequestProfile, filtered_agents: list[tuple[str, float]]) -> list[dict]:
        pool_size = self._pool_size_for(request)

        selected = filtered_agents[:pool_size]

        should_add_fallbacks = (
            not selected
            or request.confidence < 0.5
            or len(request.task_types) > 1
            or len(request.topics) > 1
            or request.constraints.get("requires_external_action")
        )

        if should_add_fallbacks:
            selected = self._add_fallbacks(selected, pool_size)

        return [self._compact_descriptor(agent_name, score) for agent_name, score in selected]

    def _pool_size_for(self, request: RequestProfile) -> int:
        if request.confidence < 0.5:
            return min(8, max(self.default_size, 6))
        return self.default_size

    def _add_fallbacks(self, selected: list[tuple[str, float]], pool_size: int) -> list[tuple[str, float]]:
        selected_names = {name for name, _ in selected}

        for agent_name in self.agent_features:
            if agent_name not in selected_names:
                selected.append((agent_name, 0.0))
                selected_names.add(agent_name)

            if len(selected) >= pool_size:
                break

        return selected

    def _compact_descriptor(self, agent_name: str, score: float) -> dict:
        feature = self.agent_features[agent_name]

        return {
            "agent_name": agent_name,
            "description": feature.description,
            "score": score,
            "tree_keys": feature.tree_keys,
            "signals": sorted(feature.graph_signals),
            "input_modes": feature.constraints.get("input_modes", []),
            "output_modes": feature.constraints.get("output_modes", []),
        }