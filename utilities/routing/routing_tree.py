from utilities.routing.agent_indexer import AgentFeature
from utilities.routing.request_normalizer import RequestProfile


class HierarchicalRoutingTree:
    def __init__(self, agent_features: dict[str, AgentFeature]):
        self.agent_features = agent_features
        self.tree = {}
        self._build_tree()

    def _build_tree(self):
        for agent_name, feature in self.agent_features.items():
            domain = feature.tree_keys.get("domain", "general")
            task_type = feature.tree_keys.get("task_type", "answer")
            modality = feature.tree_keys.get("modality", "text_to_text")
            specialty = feature.tree_keys.get("specialty", "general")

            self.tree \
                .setdefault(domain, {}) \
                .setdefault(task_type, {}) \
                .setdefault(modality, {}) \
                .setdefault(specialty, []) \
                .append(agent_name)

    def route(self, request: RequestProfile) -> list[str]:
        candidates = set()

        request_modality = self._request_modality(request)

        # 1. Caminho mais específico:
        # domain -> task_type -> modality -> specialty
        for domain in request.domains:
            domain_node = self.tree.get(domain)
            if not domain_node:
                continue

            for task_type in request.task_types:
                task_node = domain_node.get(task_type)
                if not task_node:
                    continue

                modality_node = task_node.get(request_modality)

                if modality_node:
                    self._collect_matching_specialties(
                        modality_node,
                        request,
                        candidates,
                    )

        if candidates:
            return sorted(candidates)

        # 2. Relaxa modalidade:
        # domain -> task_type -> any modality -> specialty
        for domain in request.domains:
            domain_node = self.tree.get(domain)
            if not domain_node:
                continue

            for task_type in request.task_types:
                task_node = domain_node.get(task_type)
                if not task_node:
                    continue

                for modality_node in task_node.values():
                    self._collect_matching_specialties(
                        modality_node,
                        request,
                        candidates,
                    )

        if candidates:
            return sorted(candidates)

        # 3. Relaxa specialty:
        # domain -> task_type -> any modality -> all specialties
        for domain in request.domains:
            domain_node = self.tree.get(domain)
            if not domain_node:
                continue

            for task_type in request.task_types:
                task_node = domain_node.get(task_type)
                if task_node:
                    self._collect_all(task_node, candidates)

        if candidates:
            return sorted(candidates)

        # 4. Quando o domínio da query não foi reconhecido,
        # usa sinais/tópicos para encontrar agentes compatíveis.
        self._fallback_by_signals(request, candidates)

        if candidates:
            return sorted(candidates)

        # 5. Último recurso controlado:
        # retorna poucos agentes, não todos.
        return self._controlled_global_fallback(limit=25)

    def _request_modality(self, request: RequestProfile) -> str:
        if request.input_type == "document":
            input_type = "document"
        elif request.input_type == "json":
            input_type = "json"
        else:
            input_type = "text"

        output_type = (
            "structured"
            if request.output_type in ["json", "table"]
            else "text"
        )

        return f"{input_type}_to_{output_type}"

    def _collect_matching_specialties(
        self,
        modality_node: dict,
        request: RequestProfile,
        candidates: set[str],
    ):
        matched = False

        query_signals = set(request.query_signals or set())
        query_topics = set(request.topics or [])

        for specialty, agents in modality_node.items():
            if (
                specialty in query_topics
                or specialty in query_signals
            ):
                candidates.update(agents)
                matched = True

        if matched:
            return

    def _fallback_by_signals(
        self,
        request: RequestProfile,
        candidates: set[str],
    ):
        query_signals = set(request.query_signals or set())
        query_topics = set(request.topics or [])
        request_terms = query_signals | query_topics

        if not request_terms:
            return

        scored = []

        for agent_name, feature in self.agent_features.items():
            agent_signals = set(feature.graph_signals or set())
            tree_values = set(feature.tree_keys.values())

            overlap = len(
                request_terms & (agent_signals | tree_values)
            )

            # Reforça compatibilidade de task/domain se existir
            if feature.tree_keys.get("task_type") in request.task_types:
                overlap += 2

            if feature.tree_keys.get("domain") in request.domains:
                overlap += 2

            if overlap > 0:
                scored.append((agent_name, overlap))

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # limite intencional para evitar retornar o catálogo inteiro
        for agent_name, _ in scored[:100]:
            candidates.add(agent_name)

    def _controlled_global_fallback(self, limit: int = 25) -> list[str]:
        return sorted(list(self.agent_features.keys()))[:limit]

    def _collect_all(self, node, candidates: set[str]):
        if isinstance(node, list):
            candidates.update(node)
            return

        if isinstance(node, dict):
            for child in node.values():
                self._collect_all(child, candidates)