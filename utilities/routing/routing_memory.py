import json
import os
from typing import Any


class RoutingMemory:
    def __init__(self, path: str = "utilities/routing/routing_memory.json"):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            return {
                "agent_stats": {},
                "signal_agent_success": {},
                "signal_co_occurrence": {}
            }

        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get_agent_reputation(self, agent_name: str) -> float:
        stats = self.data["agent_stats"].get(agent_name, {})
        success = stats.get("success", 0)
        total = stats.get("total", 0)

        if total == 0:
            return 0.0

        return success / total

    def get_signal_success(self, signal: str, agent_name: str) -> float:
        key = f"{signal}::{agent_name}"
        stats = self.data["signal_agent_success"].get(key, {})
        success = stats.get("success", 0)
        total = stats.get("total", 0)

        if total == 0:
            return 0.0

        return success / total

    def get_co_occurrence_boost(self, query_signals: set[str], agent_signals: set[str]) -> float:
        boost = 0.0
        shared = sorted(query_signals.intersection(agent_signals))

        for i in range(len(shared)):
            for j in range(i + 1, len(shared)):
                key = f"{shared[i]}::{shared[j]}"
                boost += self.data["signal_co_occurrence"].get(key, 0) * 0.1

        return boost

    def record_result(
        self,
        agent_name: str,
        query_signals: set[str],
        success: bool
    ):
        agent_stats = self.data["agent_stats"].setdefault(
            agent_name,
            {"success": 0, "total": 0}
        )

        agent_stats["total"] += 1
        if success:
            agent_stats["success"] += 1

        for signal in query_signals:
            key = f"{signal}::{agent_name}"
            stats = self.data["signal_agent_success"].setdefault(
                key,
                {"success": 0, "total": 0}
            )
            stats["total"] += 1
            if success:
                stats["success"] += 1

        signals = sorted(query_signals)
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                key = f"{signals[i]}::{signals[j]}"
                self.data["signal_co_occurrence"][key] = (
                    self.data["signal_co_occurrence"].get(key, 0) + 1
                )

        self.save()