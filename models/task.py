# =============================================================================
# models/task.py
# =============================================================================
# Propósito:
# Este módulo define os **modelos relacionados a tarefas** usados no protocolo Agent2Agent.
#
# Estes modelos representam:
# - Como é uma tarefa (`Task`)
# - O estado da tarefa (`TaskStatus`, `TaskState`)
# - As mensagens trocadas durante uma tarefa (`Message`, `TextPart`)
# - Parâmetros usados ao enviar, consultar ou cancelar tarefas
# =============================================================================

# -----------------------------------------------------------------------------
# Importações
# -----------------------------------------------------------------------------

from enum import Enum                 # Usado para criar constantes de valor fixo (ex: estados da tarefa)
from uuid import uuid4                # Para gerar identificadores únicos
from pydantic import BaseModel, Field   # Pydantic para validação de dados estruturados
from typing import Any, Literal, List   # Dicas de tipo (type hints) para flexibilidade e estrutura
from datetime import datetime           # Para armazenar timestamps (data e hora)


# -----------------------------------------------------------------------------
# Parte da Mensagem (Message Part): Atualmente, apenas texto é suportado
# -----------------------------------------------------------------------------

# Representa uma parte de uma mensagem, atualmente apenas texto puro é permitido
class TextPart(BaseModel):
    type: Literal["text"] = "text"  # Campo de valor fixo para identificar como tipo "text"
    text: str                       # O conteúdo de texto real (ex: "Que horas são?")

# Alias: Por enquanto, "Part" (Parte) é o mesmo que TextPart
# (usado para facilitar a refatoração no futuro, se adicionarmos ImagePart, etc.)
Part = TextPart


# -----------------------------------------------------------------------------
# Mensagem (Message): Uma entrada no histórico da tarefa
# -----------------------------------------------------------------------------

# Uma mensagem no contexto de uma tarefa, vinda do usuário ou do agente
class Message(BaseModel):
    role: Literal["user", "agent"]  # Quem enviou a mensagem: "user" ou "agent"
    parts: List[Part]               # Mensagens podem ter múltiplas partes (ex: múltiplas linhas de texto)


# -----------------------------------------------------------------------------
# TaskStatus: Descreve o estado de uma tarefa em um dado momento
# -----------------------------------------------------------------------------

class TaskStatus(BaseModel):
    state: str  # Uma string como "submitted", "working", etc. (definida mais precisamente em TaskState)
    
    # Captura automaticamente a data/hora em que o status é registrado
    timestamp: datetime = Field(default_factory=datetime.now)


# -----------------------------------------------------------------------------
# Task (Tarefa): A unidade central de trabalho no protocolo Agent2Agent
# -----------------------------------------------------------------------------

class Task(BaseModel):
    id: str
    status: TaskStatus
    history: List[Message]
    metadata: dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# Modelos de Parâmetros para Requisições API
# -----------------------------------------------------------------------------

# Usado para identificar uma tarefa, ex: ao cancelar ou consultar
class TaskIdParams(BaseModel):
    id: str                             # O ID da tarefa
    metadata: dict[str, Any] | None = None # Metadados opcionais para contexto adicional (ex: quem a submeteu)


# Estende TaskIdParams para incluir um comprimento de histórico opcional
# Útil ao consultar uma tarefa e controlar o quanto do passado você quer receber
class TaskQueryParams(TaskIdParams):
    historyLength: int | None = None      # Limita o número de mensagens retornadas no histórico da tarefa


# Parâmetros necessários para enviar uma nova tarefa a um agente
class TaskSendParams(BaseModel):
    id: str                             # ID da Tarefa (geralmente gerado no lado do cliente)
    
    # ID de Sessão usado para agrupar tarefas relacionadas (autogerado se não for fornecido)
    sessionId: str = Field(default_factory=lambda: uuid4().hex)

    message: Message                    # A mensagem que inicia a tarefa
    historyLength: int | None = None      # Comprimento do histórico opcional para retornar
    metadata: dict[str, Any] | None = None # Informações extras opcionais (ex: função do usuário, prioridade)


# -----------------------------------------------------------------------------
# TaskState: Enum para estados predefinidos do ciclo de vida da tarefa
# -----------------------------------------------------------------------------

# Enum fornece um vocabulário controlado para os estados da tarefa
class TaskState(str, Enum):
    SUBMITTED = "submitted"         # Tarefa foi recebida
    WORKING = "working"             # Tarefa está em progresso
    INPUT_REQUIRED = "input-required" # Agente está esperando por mais entrada
    COMPLETED = "completed"         # Tarefa está concluída
    CANCELED = "canceled"           # Tarefa foi cancelada pelo usuário ou sistema
    FAILED = "failed"               # Algo deu errado
    UNKNOWN = "unknown"             # Fallback (contingência) para estados indefinidos ou não reconhecidos