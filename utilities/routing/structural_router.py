import logging
from typing import Any

from models.agent import AgentCard

from utilities.routing.agent_indexer import AgentIndexer
from utilities.routing.request_normalizer import RequestNormalizer
from utilities.routing.routing_tree import HierarchicalRoutingTree
from utilities.routing.signal_graph import SignalGraphRefiner
from utilities.routing.constraint_filter import ConstraintFilter
from utilities.routing.pool_builder import CompactPoolBuilder

logger = logging.getLogger(__name__)


class StructuralAgentRouter:
    def __init__(self, agent_cards: list[AgentCard], pool_size: int = 5):
        self.agent_cards = agent_cards or []
        self.pool_size = pool_size

        self.indexer = AgentIndexer()
        self.agent_features = self.indexer.build_features(self.agent_cards)

        self.normalizer = RequestNormalizer()
        self.routing_tree = HierarchicalRoutingTree(self.agent_features)
        self.signal_graph = SignalGraphRefiner(self.agent_features)
        self.constraint_filter = ConstraintFilter(self.agent_features)
        self.pool_builder = CompactPoolBuilder(
            self.agent_features,
            default_size=self.pool_size
        )

        logger.info(
            "StructuralAgentRouter inicializado com %s agentes indexados.",
            len(self.agent_features),
        )

    def route(self, query: str) -> list[dict[str, Any]]:
        """
        Retorna apenas o compact pool final.
        Usado pelo orquestrador para montar o prompt da LLM.
        """
        trace = self.route_with_trace(query)
        return trace["compact_pool"]

    def route_with_trace(self, query: str) -> dict[str, Any]:
        """
        Executa o pipeline completo e retorna o rastreamento detalhado:

        Query
        -> Request Normalizer
        -> Hierarchical Routing Tree
        -> Signal Graph Refinement
        -> Constraint Filter
        -> Compact Pool
        """
        request = self.normalizer.normalize(query)
        self._last_query_signals = request.query_signals
        
        tree_candidates = self.routing_tree.route(request)
        logger.info("[StructuralRouter] Tree candidates: %s", tree_candidates)

        graph_ranked = self.signal_graph.refine(request, tree_candidates)
        logger.info("[StructuralRouter] Graph ranked: %s", graph_ranked)

        filtered = self.constraint_filter.filter(request, graph_ranked)
        logger.info("[StructuralRouter] Filtered: %s", filtered)

        pool = self.pool_builder.build(request, filtered)
        logger.info("[StructuralRouter] Compact pool: %s", pool)

        return {
            "request_profile": {
                "raw_query": request.raw_query,
                "normalized_query": request.normalized_query,
                "domains": request.domains,
                "task_types": request.task_types,
                "input_type": request.input_type,
                "output_type": request.output_type,
                "topics": request.topics,
                "constraints": request.constraints,
                "query_signals": sorted(request.query_signals),
                "confidence": request.confidence,
            },
            "tree_candidates": self._format_tree_candidates(tree_candidates),
            "graph_ranked": self._format_ranked_agents(graph_ranked),
            "constraint_filtered": self._format_ranked_agents(filtered),
            "compact_pool": pool,
            "summary": {
                "total_agents_indexed": len(self.agent_features),
                "tree_candidate_count": len(tree_candidates),
                "graph_ranked_count": len(graph_ranked),
                "constraint_filtered_count": len(filtered),
                "compact_pool_count": len(pool),
            },
        }

    def format_pool_for_prompt(self, query: str) -> str:
        """
        Gera o texto compacto enviado para a LLM.
        """
        pool = self.route(query)
        return self.format_pool_items_for_prompt(pool)

    def format_pool_items_for_prompt(self, pool: list[dict[str, Any]]) -> str:
        """
        Formata um compact pool já calculado.
        Útil para evitar rodar o roteador duas vezes.
        """
        if not pool:
            return "  - Nenhum agente relevante encontrado."

        lines = []

        for item in pool:
            lines.append(
                f"  - '{item['agent_name']}': {item['description']} "
                f"(score={item['score']}, route={item['tree_keys']})"
            )

        return "\n".join(lines)

    def _format_tree_candidates(self, candidates: list[str]) -> list[dict[str, Any]]:
        """
        Enriquece os candidatos da árvore com metadados úteis para UI/debug.
        """
        formatted = []

        for agent_name in candidates:
            feature = self.agent_features.get(agent_name)

            if not feature:
                formatted.append({
                    "agent_name": agent_name,
                    "tree_keys": {},
                    "description": None,
                })
                continue

            formatted.append({
                "agent_name": agent_name,
                "description": feature.description,
                "tree_keys": feature.tree_keys,
                "signals": sorted(feature.graph_signals),
                "constraints": feature.constraints,
            })

        return formatted

    def _format_ranked_agents(
        self,
        ranked_agents: list[tuple[str, float]]
    ) -> list[dict[str, Any]]:
        """
        Converte listas do tipo [(agent_name, score)] em objetos ricos para UI/debug.
        """
        formatted = []

        for agent_name, score in ranked_agents:
            feature = self.agent_features.get(agent_name)

            if not feature:
                formatted.append({
                    "agent_name": agent_name,
                    "score": score,
                    "tree_keys": {},
                    "description": None,
                })
                continue

            score_breakdown = self.signal_graph.score_breakdown(
                query_signals=getattr(self, "_last_query_signals", set()),
                agent_signals=feature.graph_signals,
                agent_name=agent_name,
            )

            formatted.append({
                "agent_name": agent_name,
                "score": score,
                "description": feature.description,
                "tree_keys": feature.tree_keys,
                "signals": sorted(feature.graph_signals),
                "constraints": feature.constraints,
                "score_breakdown": score_breakdown,
            })

        return formatted