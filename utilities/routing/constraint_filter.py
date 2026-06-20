from utilities.routing.agent_indexer import AgentFeature
from utilities.routing.request_normalizer import RequestProfile


class ConstraintFilter:
    TIER_ORDER = {
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    def __init__(self, agent_features: dict[str, AgentFeature]):
        self.agent_features = agent_features

    def filter(
        self,
        request: RequestProfile,
        scored_agents: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        accepted = []

        for agent_name, score in scored_agents:
            feature = self.agent_features.get(agent_name)
            if not feature:
                continue

            passed, _reason = self._passes_hard_constraints(request, feature)
            if not passed:
                continue

            adjusted_score = self._apply_soft_constraints(request, feature, score)
            accepted.append((agent_name, adjusted_score))

        accepted.sort(key=lambda item: item[1], reverse=True)
        return accepted

    def _passes_hard_constraints(
        self,
        request: RequestProfile,
        feature: AgentFeature
    ) -> tuple[bool, str]:
        constraints = feature.constraints
        request_constraints = request.constraints

        input_modes = constraints.get("input_modes", ["text"])
        output_modes = constraints.get("output_modes", ["text"])
        task_types = constraints.get("task_types", ["answer"])

        # 1. Input compatibility
        if request.input_type == "document":
            if not any(mode in input_modes for mode in ["document", "pdf", "file", "text"]):
                return False, "input_incompatible"

        # 2. Output compatibility
        if request.output_type == "table":
            if not any(mode in output_modes for mode in ["table", "json", "text"]):
                return False, "output_table_incompatible"

        if request.output_type == "json":
            if not any(mode in output_modes for mode in ["json", "text"]):
                return False, "output_json_incompatible"

        # 3. Task compatibility
        if not any(task in task_types for task in request.task_types):
            return False, "task_incompatible"

        # 4. Permissions
        required_permissions = set(request_constraints.get("required_permissions", ["read"]))
        agent_permissions = set(constraints.get("permissions", ["read"]))

        if not required_permissions.issubset(agent_permissions):
            return False, "missing_permissions"

        # 5. Security level
        required_security = request_constraints.get("required_security_level", "low")
        agent_security = constraints.get("security_level", "low")

        if self._tier_value(agent_security) < self._tier_value(required_security):
            return False, "security_level_too_low"

        return True, "passed"

    def _apply_soft_constraints(
        self,
        request: RequestProfile,
        feature: AgentFeature,
        score: float
    ) -> float:
        constraints = feature.constraints
        request_constraints = request.constraints

        # Exact output boost
        output_modes = constraints.get("output_modes", [])
        if request.output_type in output_modes:
            score += 1.0

        # Latency penalty
        max_latency = request_constraints.get("max_latency_tier", "high")
        agent_latency = constraints.get("latency_tier", "medium")

        if self._tier_value(agent_latency) > self._tier_value(max_latency):
            score -= 2.0

        # Cost penalty
        max_cost = request_constraints.get("max_cost_tier", "high")
        agent_cost = constraints.get("cost_tier", "medium")

        if self._tier_value(agent_cost) > self._tier_value(max_cost):
            score -= 2.0

        # Reliability boost/penalty
        min_reliability = request_constraints.get("min_reliability_tier", "low")
        agent_reliability = constraints.get("reliability_tier", "medium")

        if self._tier_value(agent_reliability) >= self._tier_value(min_reliability):
            score += 0.5
        else:
            score -= 2.0

        return score

    def _tier_value(self, tier: str) -> int:
        return self.TIER_ORDER.get(tier, 2)