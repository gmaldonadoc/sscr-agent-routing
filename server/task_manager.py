# =============================================================================
# server/task_manager.py
# =============================================================================
# 🎯 Propósito:
# Este arquivo define como as tarefas são gerenciadas no protocolo Agente-para-Agente (A2A).
#
# ✅ Inclui:
# - Uma classe base abstrata `TaskManager` que define os métodos obrigatórios.
# - Uma classe `InMemoryTaskManager` simples que mantém tarefas temporariamente na memória.
#
# ❌ Não inclui:
# - Funcionalidade de cancelar tarefa.
# - Notificações push ou atualizações em tempo real.
# - Armazenamento persistente (como um banco de dados).
# =============================================================================


# -----------------------------------------------------------------------------
# 📚 Importações Padrão do Python
# -----------------------------------------------------------------------------

from abc import ABC, abstractmethod        # Permite definir classes base abstratas (como uma interface)
from typing import Dict                    # Dict é um tipo de dicionário para armazenar pares chave-valor
import asyncio                             # Usado aqui para locks, para lidar com segurança com concorrência (operações async)


# -----------------------------------------------------------------------------
# 📦 Importações do Projeto: Modelos de Requisição e Tarefa
# -----------------------------------------------------------------------------

from models.request import (
    SendTaskRequest, SendTaskResponse,    # Para enviar tarefas ao agente
    GetTaskRequest, GetTaskResponse       # Para consultar informações da tarefa do agente
)

from models.task import (
    Task, TaskSendParams, TaskQueryParams,  # Modelos de tarefa e entrada
    TaskStatus, TaskState, Message          # Metadados da tarefa e objetos de histórico
)


# -----------------------------------------------------------------------------
# 🧩 TaskManager (Classe Base Abstrata)
# -----------------------------------------------------------------------------

class TaskManager(ABC):
    """
    🔧 Esta é uma classe de interface base.

    Todos os Gerenciadores de Tarefas devem implementar estes dois métodos assíncronos:
    - on_send_task(): para receber e processar novas tarefas
    - on_get_task(): para buscar o status atual ou histórico de conversa de uma tarefa

    Isso garante que todas as implementações sigam uma estrutura consistente.
    """

    @abstractmethod
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """📥 Este método lidará com novas tarefas recebidas."""
        pass

    @abstractmethod
    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        """📤 Este método retornará detalhes da tarefa pelo ID da tarefa."""
        pass


# -----------------------------------------------------------------------------
# 🧠 InMemoryTaskManager
# -----------------------------------------------------------------------------

class InMemoryTaskManager(TaskManager):
    """
    🧠 Um gerenciador de tarefas simples e temporário que armazena tudo na memória (RAM).

    Ótimo para:
    - Demos
    - Desenvolvimento local
    - Interações de sessão única

    ❗ Não para produção: Os dados são perdidos quando o aplicativo para ou reinicia.
    """

    def __init__(self):
        # 🗃️ Dicionário onde chave = ID da tarefa, valor = objeto Task
        self.tasks: Dict[str, Task] = {}
        # 🔐 Lock assíncrono para garantir que duas requisições não modifiquem dados ao mesmo tempo
        self.lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # 💾 upsert_task: Cria ou atualiza uma tarefa na memória
    # -------------------------------------------------------------------------
    async def upsert_task(self, params: TaskSendParams) -> Task:
        """
        Cria uma nova tarefa se ela não existir, ou atualiza o histórico se existir.

        Args:
            params: TaskSendParams – inclui ID da tarefa, ID da sessão e mensagem

        Returns:
            Task – a tarefa recém-criada ou atualizada
        """
        async with self.lock:
            # Tenta encontrar uma tarefa existente com este ID
            task = self.tasks.get(params.id)

            if task is None:
                # Se a tarefa não existe, cria-a com status "submetida"
                task = Task(
                    id=params.id,
                    status=TaskStatus(state=TaskState.SUBMITTED),
                    history=[params.message]
                )
                self.tasks[params.id] = task
            else:
                # Se a tarefa existe, adiciona a nova mensagem ao seu histórico
                task.history.append(params.message)

            return task

    # -------------------------------------------------------------------------
    # 🚫 on_send_task: Deve ser implementado por qualquer subclasse
    # -------------------------------------------------------------------------
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Este método intencionalmente não é implementado aqui.
        Subclasses como `OrchestratorTaskManager` devem sobrescrevê-lo.

        Raises:
            NotImplementedError: se alguém tentar usá-lo diretamente
        """
        raise NotImplementedError("on_send_task() deve ser implementado na subclasse")

    # -------------------------------------------------------------------------
    # 📥 on_get_task: Busca uma tarefa pelo seu ID
    # -------------------------------------------------------------------------
    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        """
        Procura uma tarefa usando seu ID, e opcionalmente retorna apenas mensagens recentes.

        Args:
            request: Um GetTaskRequest com um ID e comprimento de histórico opcional

        Returns:
            GetTaskResponse – contém a tarefa se encontrada, ou uma mensagem de erro
        """
        async with self.lock:
            query: TaskQueryParams = request.params
            task = self.tasks.get(query.id)

            if not task:
                # Se a tarefa não for encontrada, retorna um erro estruturado
                return GetTaskResponse(id=request.id, error={"message": "Tarefa não encontrada"})

            # Opcional: Corta o histórico para mostrar apenas as últimas N mensagens
            # Faz uma cópia para não afetar a original no armazenamento
            task_copy = task.model_copy()
            if query.historyLength is not None:
                # Pega as últimas N mensagens
                task_copy.history = task_copy.history[-query.historyLength:]
            
            # Retorna a resposta com a tarefa (ou sua cópia modificada)
            return GetTaskResponse(id=request.id, result=task_copy)