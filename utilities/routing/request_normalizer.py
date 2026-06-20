from dataclasses import dataclass, field
from typing import Any
import unicodedata


@dataclass
class RequestProfile:
    raw_query: str
    normalized_query: str
    domains: list[str]
    task_types: list[str]
    input_type: str
    output_type: str
    topics: list[str]
    constraints: dict[str, Any]
    query_signals: set[str] = field(default_factory=set)
    confidence: float = 0.0


class RequestNormalizer:
    def __init__(self):
        self.domain_terms = {
            # MVP antigo
            "utility": ["hora", "time", "clock", "relógio", "hello", "hi", "olá", "oi"],
            "legal": ["contrato", "contract", "cláusula", "clause", "legal", "multa", "penalty"],
            "finance": ["finance", "financial", "revenue", "invoice", "fatura", "pagamento"],
            "coding": ["code", "python", "bug", "api", "debug", "programar"],

            # Domínios sintéticos
            "agricultura": ["agricultura", "agronegócio", "safra", "solo", "irrigação", "pragas", "cultura agrícola"],
            "educacao": ["educação", "ensino", "aluno", "professor", "aula", "quiz", "currículo", "redação"],
            "energia": ["energia", "rede elétrica", "demanda energética", "turbina", "consumo energético", "grid"],
            "ciencia_ambiental": ["ciência ambiental", "sustentabilidade", "ambiental", "água", "ar", "desmatamento", "pegada de carbono", "risco ambiental"],
            "financas_e_negocios": ["finanças", "negócios", "crédito", "fluxo de caixa", "portfólio", "auditoria fiscal", "risco financeiro"],
            "governo_e_setor_publico": ["governo", "setor público", "administração pública", "diário oficial", "licitação", "políticas públicas", "orçamento público"],
            "assistencia_medica": ["saúde", "assistência médica", "hospital", "clínica", "paciente", "prontuário", "sintomas", "medicamento", "medicamentosa"],
            "hospitalidade_e_turismo": ["hotel", "turismo", "viagem", "reservas", "hóspede", "passeios", "hospitalidade"],
            "recursos_humanos": ["recursos humanos", "rh", "currículos", "candidatos", "clima organizacional", "competências", "turnover", "vagas"],
            "fabricacao_industrial": ["manufatura", "industrial", "fábrica", "linha de produção", "máquina", "oee", "qualidade por visão"],
            "seguro": ["seguro", "seguros", "sinistro", "apólice", "atuarial", "indenização", "subscrição"],
            "juridico": ["jurídico", "legal", "jurisprudência", "petição", "processual", "processo", "contratual", "cláusulas"],
            "ciencias_da_vida": ["ciências da vida", "biotecnologia", "dna", "biomarcadores", "ensaio clínico", "expressão gênica", "compostos químicos"],
            "marketing_e_publicidade": ["marketing", "publicidade", "campanha", "copywriting", "audiência", "marca", "funil de marketing", "anúncios"],
            "midia_e_entretenimento": ["mídia", "entretenimento", "conteúdo", "vídeo", "legendagem", "engajamento"],
            "imobiliaria": ["imobiliário", "imóvel", "aluguel", "locação", "zoneamento", "lead imobiliário", "locatário"],
            "pesquisa_e_desenvolvimento": ["pesquisa", "desenvolvimento", "inovação", "patentes", "literatura científica", "experimentos", "estado da arte", "hipóteses"],
            "varejo_e_comercio_eletronico": ["varejo", "comércio eletrônico", "marketplace", "carrinho abandonado", "cross-sell", "estoque", "sac", "produtos"],
            "servicos_sociais": ["serviços sociais", "benefícios", "assistência social", "vulnerabilidade", "ong", "impacto comunitário", "atendimento prioritário"],
            "esportes_e_fitness": ["esportes", "fitness", "treino", "biométrica", "dieta", "lesões", "atleta", "recuperação"],
            "tecnologia": ["tecnologia", "software", "código", "api", "logs", "bugs", "dependências", "documentação de api"],
            "telecomunicacoes": ["telecomunicações", "conectividade", "rede", "fibra óptica", "espectro", "sinal", "tráfego de rede", "churn de rede"],
            "transporte": ["transporte", "logística", "mobilidade", "rotas", "frotas", "última milha", "veículos", "entregas", "atrasos"],
            "confianca_e_seguranca": ["segurança", "confiança", "fraude", "phishing", "vulnerabilidade", "abuso", "compliance", "risco do usuário"],
        }

        self.topic_terms = {
            # MVP antigo
            "time": ["time", "hora", "clock", "relógio"],
            "greeting": ["hello", "hi", "greet", "oi", "olá"],
            "contract": ["contract", "contrato"],
            "penalty_clause": ["penalty", "multa"],
            "file_system": ["file", "arquivo", "folder", "pasta"],
            "terminal": ["terminal", "shell", "command", "comando"],

            # Skills sintéticas
            "analise_safra": ["análise de safra", "safra"],
            "telemetria_solo": ["telemetria do solo", "solo"],
            "previsao_pragas": ["previsão de pragas", "pragas"],
            "otimizacao_irrigacao": ["otimização de irrigação", "irrigação"],
            "monitoramento_umidade": ["monitoramento de umidade", "umidade"],
            "classificacao_cultura": ["classificação de culturas agrícolas", "culturas agrícolas"],

            "avaliacao_desempenho": ["avaliação de desempenho", "desempenho"],
            "geracao_quiz": ["geração de questionários", "questionários", "quiz"],
            "tutoria_personalizada": ["tutoria personalizada"],
            "adaptacao_curricular": ["adaptação curricular"],
            "correcao_redacao": ["correção de redações", "redações"],
            "planejamento_aula": ["planejamento de aulas", "aulas"],

            "predicao_demanda": ["predição de demanda energética", "demanda energética"],
            "otimizacao_grid": ["otimização da rede elétrica", "rede elétrica", "grid"],
            "manutencao_turbina": ["manutenção de turbinas", "turbinas"],
            "analise_eficiencia": ["análise de eficiência energética", "eficiência energética"],
            "previsao_consumo": ["previsão de consumo energético", "consumo energético"],
            "deteccao_anomalia_energia": ["detecção de anomalias energéticas", "anomalias energéticas"],

            "calculo_pegada_carbono": ["cálculo de pegada de carbono", "pegada de carbono"],
            "modelagem_climatica": ["modelagem climática"],
            "monitoramento_desmatamento": ["monitoramento de desmatamento", "desmatamento"],
            "analise_qualidade_ar": ["análise de qualidade do ar", "qualidade do ar"],
            "analise_qualidade_agua": ["análise de qualidade da água", "qualidade da água"],
            "previsao_risco_ambiental": ["previsão de risco ambiental", "risco ambiental"],

            "analise_credito": ["análise de crédito", "crédito"],
            "conciliacao_bancaria": ["conciliação bancária"],
            "auditoria_fiscal": ["auditoria fiscal"],
            "otimizacao_portfolio": ["otimização de portfólio", "portfólio"],
            "previsao_fluxo_caixa": ["previsão de fluxo de caixa", "fluxo de caixa"],
            "analise_risco_financeiro": ["análise de risco financeiro", "risco financeiro"],

            "consulta_diario_oficial": ["consulta ao diário oficial", "diário oficial"],
            "triage_protocolo": ["triagem de protocolos", "protocolos"],
            "analise_licitacao": ["análise de licitações", "licitações", "licitação"],
            "classificacao_requerimentos": ["classificação de requerimentos", "requerimentos"],
            "monitoramento_politicas_publicas": ["monitoramento de políticas públicas", "políticas públicas"],
            "analise_orcamento_publico": ["análise de orçamento público", "orçamento público"],

            "triagem_sintomas": ["triagem de sintomas", "sintomas"],
            "analise_prontuario": ["análise de prontuários médicos", "prontuários", "prontuário"],
            "checagem_interacao_medicamentosa": ["checagem de interação medicamentosa", "interação medicamentosa", "medicamentos"],
            "classificacao_risco_paciente": ["classificação de risco do paciente", "risco do paciente"],
            "resumo_clinico": ["resumo clínico"],
            "codificacao_procedimentos": ["codificação de procedimentos médicos", "procedimentos médicos"],

            "screening_curriculo": ["triagem de currículos", "currículos", "currículo"],
            "analise_clima_organizacional": ["análise de clima organizacional", "clima organizacional"],
            "mapeamento_competencias": ["mapeamento de competências", "competências"],
            "descricao_vagas": ["descrição de vagas", "vagas"],
            "analise_turnover": ["análise de turnover", "turnover"],
            "triagem_candidatos": ["triagem de candidatos", "candidatos"],

            "manutencao_preditiva": ["manutenção preditiva"],
            "controle_qualidade_visao": ["controle de qualidade por visão computacional", "visão computacional"],
            "otimizacao_gargalo": ["otimização de gargalos", "gargalos"],
            "monitoramento_linha_producao": ["monitoramento de linha de produção", "linha de produção"],
            "previsao_falha_maquina": ["previsão de falhas de máquinas", "falhas de máquinas"],
            "analise_oee": ["análise de oee", "oee"],

            "analise_sinistro": ["análise de sinistros", "sinistros"],
            "calculo_atuarial": ["cálculo atuarial"],
            "subscricao_risco": ["subscrição de risco"],
            "deteccao_fraude_apolice": ["detecção de fraude em apólices", "fraude em apólices"],
            "classificacao_apolice": ["classificação de apólices", "apólices"],
            "estimativa_indenizacao": ["estimativa de indenização", "indenização"],

            "analise_contratual": ["análise contratual"],
            "revisao_jurisprudencia": ["revisão de jurisprudência", "jurisprudência"],
            "redacao_peticao": ["redação de petições", "petições"],
            "extracao_dados_processuais": ["extração de dados processuais", "dados processuais"],
            "analise_clausula_multa": ["análise de cláusulas de multa", "cláusulas de multa"],
            "classificacao_risco_legal": ["classificação de risco jurídico", "risco jurídico"],

            "alinhamento_sequencia_dna": ["alinhamento de sequências de dna", "sequências de dna", "dna"],
            "triagem_compostos_quimicos": ["triagem de compostos químicos", "compostos químicos"],
            "analise_ensaio_clinico": ["análise de ensaios clínicos", "ensaios clínicos"],
            "classificacao_biomarcadores": ["classificação de biomarcadores", "biomarcadores"],
            "analise_expressao_genica": ["análise de expressão gênica", "expressão gênica"],
            "revisao_literatura_biomedica": ["revisão de literatura biomédica", "literatura biomédica"],

            "geracao_copywriting": ["geração de copywriting", "copywriting"],
            "analise_sentimento_marca": ["análise de sentimento de marca", "sentimento de marca"],
            "otimizacao_bid_ads": ["otimização de lances de anúncios", "lances de anúncios"],
            "segmentacao_audiencia": ["segmentação de audiência", "audiência"],
            "geracao_campanha": ["geração de campanhas", "campanhas"],
            "analise_funil_marketing": ["análise de funil de marketing", "funil de marketing"],

            "recomendacao_conteudo": ["recomendação de conteúdo", "conteúdo"],
            "legendagem_automatica": ["legendagem automática"],
            "analise_engajamento": ["análise de engajamento", "engajamento"],
            "classificacao_video": ["classificação de vídeos", "vídeos"],
            "roteirizacao_conteudo": ["roteirização de conteúdo"],
            "moderacao_midia": ["moderação de mídia"],

            "avaliacao_imovel": ["avaliação de imóveis", "imóveis", "imóvel"],
            "analise_zoneamento": ["análise de zoneamento", "zoneamento"],
            "geracao_contrato_locacao": ["geração de contratos de locação", "contratos de locação"],
            "estimativa_aluguel": ["estimativa de aluguel", "aluguel"],
            "classificacao_lead_imobiliario": ["classificação de leads imobiliários", "leads imobiliários"],
            "analise_risco_locatario": ["análise de risco de locatários", "risco de locatários"],

            "busca_patentes": ["busca de patentes", "patentes"],
            "sintese_literatura_cientifica": ["síntese de literatura científica", "literatura científica"],
            "design_experimentos": ["desenho de experimentos", "experimentos"],
            "analise_estado_da_arte": ["análise do estado da arte", "estado da arte"],
            "geracao_hipoteses": ["geração de hipóteses", "hipóteses"],
            "avaliacao_viabilidade_tecnica": ["avaliação de viabilidade técnica", "viabilidade técnica"],

            "sugestao_cross_sell": ["sugestão de cross-sell", "cross-sell"],
            "gestao_estoque_preditiva": ["gestão preditiva de estoque", "estoque"],
            "sac_reclamacoes": ["atendimento de reclamações de sac", "reclamações de sac", "sac"],
            "precificacao_dinamica": ["precificação dinâmica"],
            "recomendacao_produtos": ["recomendação de produtos", "produtos"],
            "analise_carrinho_abandonado": ["análise de carrinho abandonado", "carrinho abandonado"],

            "mapeamento_vulnerabilidade": ["mapeamento de vulnerabilidade social", "vulnerabilidade social"],
            "triagem_beneficios": ["triagem de benefícios", "benefícios"],
            "alocacao_recursos_comunidade": ["alocação de recursos comunitários", "recursos comunitários"],
            "analise_caso_social": ["análise de casos sociais", "casos sociais"],
            "priorizacao_atendimento": ["priorização de atendimento"],
            "monitoramento_impacto_social": ["monitoramento de impacto social", "impacto social"],

            "planejamento_treino": ["planejamento de treinos", "treinos"],
            "analise_biometrica": ["análise biométrica", "biométrica"],
            "otimizacao_dieta": ["otimização de dieta", "dieta"],
            "prevencao_lesoes": ["prevenção de lesões", "lesões"],
            "analise_performance_atleta": ["análise de performance de atletas", "performance de atletas"],
            "recomendacao_recuperacao": ["recomendação de recuperação", "recuperação"],

            "geracao_codigo": ["geração de código", "código"],
            "refatoracao_automatica": ["refatoração automática"],
            "analise_logs": ["análise de logs", "logs"],
            "geracao_documentacao_api": ["geração de documentação de api", "documentação de api"],
            "deteccao_bug": ["detecção de bugs", "bugs"],
            "analise_dependencias": ["análise de dependências", "dependências"],

            "analise_churn_rede": ["análise de churn de rede", "churn de rede"],
            "otimizacao_espectro": ["otimização de espectro", "espectro"],
            "diagnostico_falha_fibra": ["diagnóstico de falhas em fibra óptica", "fibra óptica"],
            "previsao_trafego_rede": ["previsão de tráfego de rede", "tráfego de rede"],
            "classificacao_ticket_suporte": ["classificação de tickets de suporte", "tickets de suporte"],
            "analise_qualidade_sinal": ["análise de qualidade de sinal", "qualidade de sinal"],

            "otimizacao_rotas_frotas": ["otimização de rotas de frotas", "rotas de frotas"],
            "calculo_tempo_estimado": ["cálculo de tempo estimado", "tempo estimado"],
            "roteirizacao_ultima_milha": ["roteirização de última milha", "última milha"],
            "previsao_atrasos": ["previsão de atrasos", "atrasos"],
            "alocacao_veiculos": ["alocação de veículos", "veículos"],
            "analise_consumo_combustivel": ["análise de consumo de combustível", "combustível"],

            "deteccao_fraude": ["detecção de fraude", "fraude"],
            "moderacao_conteudo": ["moderação de conteúdo"],
            "analise_vulnerabilidade_codigo": ["análise de vulnerabilidades de código", "vulnerabilidades de código"],
            "anti_phishing": ["detecção de phishing", "phishing"],
            "classificacao_risco_usuario": ["classificação de risco do usuário", "risco do usuário"],
            "deteccao_abuso_plataforma": ["detecção de abuso de plataforma", "abuso de plataforma"],
        }

    def normalize(self, query: str) -> RequestProfile:
        text = query.lower().strip()
        norm_text = self._normalize_text(text)

        domains = self._extract_domains(norm_text)
        task_types = self._extract_task_types(norm_text)
        input_type = self._infer_input_type(norm_text)
        output_type = self._infer_output_type(norm_text)
        topics = self._extract_topics(norm_text)
        constraints = self._extract_constraints(norm_text)

        signals = set()
        signals.update(domains)
        signals.update(task_types)
        signals.update(topics)
        signals.add(f"{input_type}_input")
        signals.add(f"{output_type}_output")

        confidence = self._estimate_confidence(domains, task_types, topics)

        return RequestProfile(
            raw_query=query,
            normalized_query=norm_text,
            domains=domains,
            task_types=task_types,
            input_type=input_type,
            output_type=output_type,
            topics=topics,
            constraints=constraints,
            query_signals=signals,
            confidence=confidence,
        )

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = "".join(
            ch for ch in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(ch)
        )
        return text

    def _matches_any(self, text: str, terms: list[str]) -> bool:
        norm_terms = [self._normalize_text(term) for term in terms]
        return any(term in text for term in norm_terms)

    def _extract_domains(self, text: str) -> list[str]:
        domains = []

        for domain, terms in self.domain_terms.items():
            if self._matches_any(text, terms):
                domains.append(domain)

        return domains or ["general"]

    def _extract_task_types(self, text: str) -> list[str]:
        tasks = []

        task_patterns = {
            "greet": ["oi", "ola", "hello", "hi", "greet", "saudar"],
            "answer": ["que horas", "hora", "time", "clock", "explique", "explicar", "duvida", "pode me explicar"],
            "extract": ["extraia", "extrair", "extract", "liste", "list", "identifique e extraia", "transforme os dados"],
            "analyze": ["analise", "analise este caso", "avaliar", "avalie", "assess", "faca uma analise"],
            "summarize": ["resuma", "summarize", "resumo", "summary", "sintetize"],
            "classify": ["classifique", "classificar", "classify", "organize estes casos", "categoria"],
            "generate": ["gere", "gerar", "crie", "prepare", "relatorio", "recomendacao"],
            "validate": ["valide", "validar", "verifique", "confira", "criterios", "inconsistencias"],
            "execute": ["execute", "executar", "rode", "run", "terminal", "shell", "realize", "aplique"],
        }

        for task_type, terms in task_patterns.items():
            if self._matches_any(text, terms):
                tasks.append(task_type)

        return tasks or ["answer"]

    def _extract_topics(self, text: str) -> list[str]:
        topics = []

        for topic, terms in self.topic_terms.items():
            if self._matches_any(text, terms):
                topics.append(topic)

        return topics

    def _infer_input_type(self, text: str) -> str:
        file_creation_terms = [
            "escreva em um arquivo",
            "escreva a",
            "salve em um arquivo",
            "salvar em um arquivo",
            "grave em um arquivo",
            "gravar em um arquivo",
            "crie um arquivo",
            "criar um arquivo",
            "write to file",
            "save to file",
            "create file",
        ]

        if self._matches_any(text, file_creation_terms):
            return "text"

        document_input_terms = [
            "pdf",
            "documento",
            "document",
            "arquivo enviado",
            "analise este arquivo",
            "analise o arquivo",
            "leia este arquivo",
            "ler arquivo",
            "contrato",
            "contract",
            "documentos",
        ]

        if self._matches_any(text, document_input_terms):
            return "document"

        if "json" in text:
            return "json"

        return "text"

    def _infer_output_type(self, text: str) -> str:
        if self._matches_any(text, ["tabela", "table", "formato de tabela"]):
            return "table"
        if "json" in text:
            return "json"
        return "text"

    def _extract_constraints(self, text: str) -> dict[str, Any]:
        creates_file = self._matches_any(text, [
            "escreva em um arquivo",
            "escreva a saudação em um arquivo",
            "salve em um arquivo",
            "salvar em um arquivo",
            "grave em um arquivo",
            "gravar em um arquivo",
            "crie um arquivo",
            "criar um arquivo",
            "write to file",
            "save to file",
            "create file",
        ])

        requires_document = (
            not creates_file
            and self._matches_any(text, [
                "pdf",
                "documento",
                "document",
                "arquivo enviado",
                "analise este arquivo",
                "analise o arquivo",
                "leia este arquivo",
                "ler arquivo",
                "contrato",
                "contract",
                "documentos",
            ])
        )

        requires_external_action = self._matches_any(text, [
            "execute",
            "executar",
            "rode",
            "run",
            "terminal",
            "shell",
            "comando",
            "command",
            "salve",
            "salvar",
            "grave",
            "gravar",
            "escreva em um arquivo",
            "escreva a saudação em um arquivo",
            "crie um arquivo",
            "run_shell_command",
        ])

        return {
            "requires_table": self._matches_any(text, ["tabela", "table", "formato de tabela"]),
            "requires_json": "json" in text,
            "requires_document": requires_document,
            "requires_external_action": requires_external_action,
            "creates_file": creates_file,
            "high_priority": self._matches_any(text, ["urgente", "urgent", "rápido", "quickly"]),

            "max_latency_tier": self._infer_max_latency_tier(text),
            "max_cost_tier": self._infer_max_cost_tier(text),
            "min_reliability_tier": self._infer_min_reliability_tier(text),
        }

    def _infer_max_latency_tier(self, text: str) -> str:
        if self._matches_any(text, ["urgente", "rapido", "imediato", "quickly", "asap"]):
            return "low"
        return "high"

    def _infer_max_cost_tier(self, text: str) -> str:
        if self._matches_any(text, ["barato", "baixo custo", "low cost", "cheap"]):
            return "low"
        return "high"

    def _infer_min_reliability_tier(self, text: str) -> str:
        if self._matches_any(text, ["confiavel", "preciso", "critico", "critical", "reliable"]):
            return "high"
        return "low"

    def _estimate_confidence(self, domains: list[str], tasks: list[str], topics: list[str]) -> float:
        score = 0.0

        if domains and domains != ["general"]:
            score += 0.35
        if tasks and tasks != ["answer"]:
            score += 0.35
        if topics:
            score += 0.30

        return round(min(score, 1.0), 2)