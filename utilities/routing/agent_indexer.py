from dataclasses import dataclass
from typing import Any

from models.agent import AgentCard


@dataclass
class AgentFeature:
    agent_name: str
    agent_url: str
    description: str
    source_text: str
    tree_keys: dict[str, str]
    graph_signals: set[str]
    constraints: dict[str, Any]


class AgentIndexer:
    def __init__(self):
        self.known_domains = {
            # Domínios originais do MVP
            "utility",
            "legal",
            "finance",
            "coding",

            # Domínios sintéticos OASF
            "agricultura",
            "educacao",
            "energia",
            "ciencia_ambiental",
            "financas_e_negocios",
            "governo_e_setor_publico",
            "assistencia_medica",
            "hospitalidade_e_turismo",
            "recursos_humanos",
            "fabricacao_industrial",
            "seguro",
            "juridico",
            "ciencias_da_vida",
            "marketing_e_publicidade",
            "midia_e_entretenimento",
            "imobiliaria",
            "pesquisa_e_desenvolvimento",
            "varejo_e_comercio_eletronico",
            "servicos_sociais",
            "esportes_e_fitness",
            "tecnologia",
            "telecomunicacoes",
            "transporte",
            "confianca_e_seguranca",
        }

        self.known_tasks = {
            "answer",
            "greet",
            "extract",
            "analyze",
            "summarize",
            "classify",
            "generate",
            "validate",
            "execute",
        }

        self.known_input_tags = {
            "text_input",
            "document_input",
            "json_input",
            "pdf_input",
            "file_input",
        }

        self.known_output_tags = {
            "text_output",
            "json_output",
            "table_output",
            "document_output",
            "file_output",
        }

        self.known_operational_tags = {
            "latency_low",
            "latency_medium",
            "latency_high",
            "cost_low",
            "cost_medium",
            "cost_high",
            "reliability_low",
            "reliability_medium",
            "reliability_high",
        }

        self.framework_tags = {
            "langgraph",
            "autogen",
            "crewai",
            "custom_agent_v2",
            "semantic_kernel",
        }

        self.domain_terms = {
            "utility": [
                "time",
                "hora",
                "clock",
                "relógio",
                "hello",
                "greet",
                "greeting",
                "saudar",
            ],
            "legal": [
                "contract",
                "contrato",
                "clause",
                "cláusula",
                "legal",
                "penalty",
                "multa",
            ],
            "finance": [
                "invoice",
                "revenue",
                "finance",
                "financial",
                "pagamento",
                "fatura",
            ],
            "coding": [
                "code",
                "python",
                "bug",
                "api",
                "debug",
                "programar",
            ],
        }

        self.task_terms = {
            "answer": [
                "tell",
                "return",
                "what",
                "qual",
                "responda",
                "diga",
                "explique",
                "explicar",
                "dúvida",
            ],
            "greet": [
                "hello",
                "hi",
                "greet",
                "olá",
                "oi",
                "saudar",
            ],
            "extract": [
                "extract",
                "extraia",
                "extrair",
                "list",
                "liste",
                "identifique",
            ],
            "analyze": [
                "analyze",
                "analise",
                "análise",
                "avaliar",
                "avalie",
                "assess",
            ],
            "summarize": [
                "summarize",
                "resuma",
                "summary",
                "resumo",
                "sintetize",
            ],
            "classify": [
                "classifique",
                "classificar",
                "classify",
                "categoria",
                "categorize",
            ],
            "generate": [
                "generate",
                "gerar",
                "gere",
                "crie",
                "prepare",
                "relatório",
                "recomendação",
            ],
            "validate": [
                "validate",
                "validar",
                "valide",
                "verifique",
                "confira",
                "critérios",
            ],
            "execute": [
                "run",
                "execute",
                "executar",
                "terminal",
                "shell",
                "realize",
                "aplique",
            ],
        }

    def build_features(self, agent_cards: list[AgentCard]) -> dict[str, AgentFeature]:
        features = {}

        for card in agent_cards:
            feature = self.index_card(card)
            features[feature.agent_name] = feature

        return features

    def index_card(self, card: AgentCard) -> AgentFeature:
        source_text = self._build_source_text(card)
        tags = self._collect_tags(card)

        input_modes = self._collect_input_modes(card, tags)
        output_modes = self._collect_output_modes(card, tags)

        domains = self._extract_domains_from_tags(tags)
        if not domains:
            domains = self._extract_domains(source_text)

        task_types = self._extract_task_types_from_tags(tags)
        if not task_types:
            task_types = self._extract_task_types(source_text)

        topics = self._extract_topics_from_tags(tags)
        if not topics:
            topics = self._extract_topics(source_text)

        domain = domains[0] if domains else "general"
        task_type = task_types[0] if task_types else "answer"
        specialty = topics[0] if topics else "general"
        modality = self._infer_modality(input_modes, output_modes)

        graph_signals = set()
        graph_signals.update(domains)
        graph_signals.update(task_types)
        graph_signals.update(topics)
        graph_signals.update(tags)
        graph_signals.update(self._normalize_modes(input_modes, suffix="_input"))
        graph_signals.update(self._normalize_modes(output_modes, suffix="_output"))

        return AgentFeature(
            agent_name=card.name,
            agent_url=card.url,
            description=card.description,
            source_text=source_text,
            tree_keys={
                "domain": domain,
                "task_type": task_type,
                "modality": modality,
                "specialty": specialty,
            },
            graph_signals=graph_signals,
            constraints={
                "input_modes": input_modes,
                "output_modes": output_modes,
                "task_types": task_types,
                "domains": domains,
                "topics": topics,

                "latency_tier": self._extract_tier_from_tags(tags, "latency")
                or self._infer_latency_tier(source_text),

                "cost_tier": self._extract_tier_from_tags(tags, "cost")
                or self._infer_cost_tier(source_text),

                "reliability_tier": self._extract_tier_from_tags(tags, "reliability")
                or self._infer_reliability_tier(source_text),

                "security_level": self._infer_security_level(source_text),
                "permissions": self._infer_permissions(source_text),
            },
        )

    def _build_source_text(self, card: AgentCard) -> str:
        parts = [card.name, card.description]

        for skill in card.skills or []:
            parts.append(skill.id)
            parts.append(skill.name)
            if skill.description:
                parts.append(skill.description)
            if skill.tags:
                parts.extend(skill.tags)
            if skill.examples:
                parts.extend(skill.examples)

        return " ".join(parts).lower()

    def _collect_tags(self, card: AgentCard) -> set[str]:
        tags = set()

        for skill in card.skills or []:
            for tag in skill.tags or []:
                tags.add(str(tag).lower())

            if skill.id:
                tags.add(skill.id.lower())

        return tags

    def _collect_input_modes(self, card: AgentCard, tags: set[str] | None = None) -> list[str]:
        modes = set()

        for skill in card.skills or []:
            for mode in skill.inputModes or []:
                modes.add(mode.lower())

        for tag in tags or set():
            if tag.endswith("_input"):
                modes.add(tag.replace("_input", ""))

        return sorted(modes) if modes else ["text"]

    def _collect_output_modes(self, card: AgentCard, tags: set[str] | None = None) -> list[str]:
        modes = set()

        for skill in card.skills or []:
            for mode in skill.outputModes or []:
                modes.add(mode.lower())

        for tag in tags or set():
            if tag.endswith("_output"):
                modes.add(tag.replace("_output", ""))

        return sorted(modes) if modes else ["text"]

    def _extract_domains_from_tags(self, tags: set[str]) -> list[str]:
        return sorted(tag for tag in tags if tag in self.known_domains)

    def _extract_task_types_from_tags(self, tags: set[str]) -> list[str]:
        return sorted(tag for tag in tags if tag in self.known_tasks)

    def _extract_topics_from_tags(self, tags: set[str]) -> list[str]:
        ignored = (
            self.known_domains
            | self.known_tasks
            | self.known_input_tags
            | self.known_output_tags
            | self.known_operational_tags
            | self.framework_tags
        )

        topics = []

        for tag in sorted(tags):
            if tag not in ignored:
                topics.append(tag)

        return topics

    def _extract_tier_from_tags(self, tags: set[str], prefix: str) -> str | None:
        for tier in ["low", "medium", "high"]:
            if f"{prefix}_{tier}" in tags:
                return tier

        return None

    def _extract_domains(self, text: str) -> list[str]:
        found = []

        for domain, terms in self.domain_terms.items():
            if any(term in text for term in terms):
                found.append(domain)

        return found or ["general"]

    def _extract_task_types(self, text: str) -> list[str]:
        found = []

        for task_type, terms in self.task_terms.items():
            if any(term in text for term in terms):
                found.append(task_type)

        return found or ["answer"]

    def _extract_topics(self, text: str) -> list[str]:
        topics = []

        topic_terms = {
            "time": ["time", "hora", "clock", "relógio"],
            "greeting": ["hello", "hi", "greet", "oi", "olá"],
            "contract": ["contract", "contrato"],
            "penalty_clause": ["penalty", "multa"],
            "file_system": ["file", "arquivo", "folder", "pasta"],
            "terminal": ["terminal", "shell", "command", "comando"],
        }

        for topic, terms in topic_terms.items():
            if any(term in text for term in terms):
                topics.append(topic)

        return topics

    def _infer_modality(self, input_modes: list[str], output_modes: list[str]) -> str:
        input_type = "text"
        output_type = "text"

        if any(mode in input_modes for mode in ["pdf", "document", "file"]):
            input_type = "document"

        if any(mode in input_modes for mode in ["json"]):
            input_type = "json"

        if any(mode in output_modes for mode in ["json", "table"]):
            output_type = "structured"

        return f"{input_type}_to_{output_type}"

    def _normalize_modes(self, modes: list[str], suffix: str) -> set[str]:
        return {f"{mode}_{suffix}".replace("__", "_") for mode in modes}

    def _infer_latency_tier(self, text: str) -> str:
        if any(w in text for w in ["fast", "rápido", "quick", "low latency", "baixa latência"]):
            return "low"
        if any(w in text for w in ["slow", "lento", "batch", "long running", "latência mais alta"]):
            return "high"
        return "medium"

    def _infer_cost_tier(self, text: str) -> str:
        if any(w in text for w in ["expensive", "costly", "premium", "high cost", "custo operacional elevado"]):
            return "high"
        if any(w in text for w in ["cheap", "low cost", "barato", "lightweight", "baixo custo operacional"]):
            return "low"
        return "medium"

    def _infer_reliability_tier(self, text: str) -> str:
        if any(w in text for w in ["stable", "production", "reliable", "confiável", "alta confiabilidade"]):
            return "high"
        if any(w in text for w in ["experimental", "beta", "unstable", "confiabilidade experimental"]):
            return "low"
        return "medium"

    def _infer_security_level(self, text: str) -> str:
        if any(w in text for w in ["admin", "terminal", "shell", "write", "execute", "executar"]):
            return "high"
        if any(w in text for w in ["file", "arquivo", "document", "documento"]):
            return "medium"
        return "low"

    def _infer_permissions(self, text: str) -> list[str]:
        permissions = ["read"]

        if any(w in text for w in ["write", "escrever", "salvar", "file", "arquivo"]):
            permissions.append("write")

        if any(w in text for w in ["execute", "executar", "terminal", "shell", "command", "comando"]):
            permissions.append("execute")

        return sorted(set(permissions))