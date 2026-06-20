# =============================================================================
# models/agent.py
# =============================================================================
# Propósito:
# Este arquivo define os modelos de dados relacionados a agentes usados em todo o
# sistema Agent2Agent (A2A).
#
# Isso inclui:
# - Capacidades (Capabilities) que um agente suporta (ex: streaming, notificações push)
# - Metadados sobre o próprio agente (AgentCard)
# - Metadados para cada habilidade (skill) que o agente pode executar (AgentSkill)
#
# Essas classes ajudam a descrever a identidade do agente, seus recursos e como
# ele interage com outros agentes ou clientes de forma estruturada e consistente.
# =============================================================================

# -----------------------------------------------------------------------------
# Importações
# -----------------------------------------------------------------------------

# BaseModel é uma classe base poderosa do Pydantic.
# Ela valida e converte automaticamente dados de entrada em tipos Python
from pydantic import BaseModel

# Dica de tipo (type hint) 'List' para declarar campos de lista
from typing import List


# -----------------------------------------------------------------------------
# AgentCapabilities (Capacidades do Agente)
# -----------------------------------------------------------------------------
# Esta classe define quais recursos ou protocolos o agente suporta.
# Essas capacidades podem ser usadas por clientes A2A ou diretórios para
# entender como interagir com o agente.
class AgentCapabilities(BaseModel):
    # Indica se o agente pode enviar resultados intermediários de tarefas via streaming
    streaming: bool = False

    # Indica se o agente pode enviar atualizações via HTTP push/webhooks
    pushNotifications: bool = False

    # Se ativado, o agente mantém um histórico das transições de estado da tarefa (ex: "iniciada", "concluída")
    # Útil para depuração (debugging) ou auditoria
    stateTransitionHistory: bool = False


# -----------------------------------------------------------------------------
# AgentSkill (Habilidade do Agente)
# -----------------------------------------------------------------------------
# Esta classe define metadados sobre uma única habilidade (skill) que o agente oferece.
# Cada habilidade corresponde a um tipo específico de tarefa que o agente pode realizar.
class AgentSkill(BaseModel):
    # Identificador único para a habilidade (ex: "get_time")
    id: str

    # Nome legível para a habilidade (ex: "Obter Hora Atual")
    name: str

    # Descrição opcional para ajudar os usuários a entender o que a habilidade faz
    description: str | None = None

    # Tags opcionais para ajudar a categorizar ou pesquisar a habilidade (ex: ["tempo", "relógio"])
    tags: List[str] | None = None

    # Frases de exemplo opcionais às quais esta habilidade pode responder
    # Útil para interfaces que sugerem consultas ao usuário
    examples: List[str] | None = None

    # Lista opcional de modos de entrada suportados (ex: ["text", "json"])
    inputModes: List[str] | None = None

    # Lista opcional de modos de saída suportados (ex: ["text", "image"])
    outputModes: List[str] | None = None

# -----------------------------------------------------------------------------
# AgentCard (Cartão do Agente)
# -----------------------------------------------------------------------------
# Esta classe fornece metadados centrais sobre um agente.
# Esta informação pode ser compartilhada com um serviço de diretório ou outros agentes
# para descrever o que o agente faz, onde alcançá-lo e quais capacidades ele suporta.
class AgentCard(BaseModel):
    # Nome legível do agente (ex: "Agente de Horário")
    name: str

    # Descrição do propósito ou caso de uso do agente
    description: str

    # URL onde o agente está hospedado (pode ser usada para enviar requisições para ele)
    url: str

    # Versão semântica do agente (ex: "1.0.0")
    version: str

    # As capacidades que este agente suporta (usa o modelo AgentCapabilities acima)
    capabilities: AgentCapabilities

    # Lista de habilidades (skills) que este agente pode realizar
    # (Note que aqui estamos aninhando a definição completa do AgentSkill)
    skills: List[AgentSkill]