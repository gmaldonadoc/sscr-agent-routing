# =============================================================================
# agentes/tell_time_agent/task_manager.py
# (Baseado no seu 'google_adk/task_manager.py')
# =============================================================================
# 🎯 Propósito:
# Este arquivo conecta seu agente (TellTimeAgent) ao sistema de
# gerenciamento de tarefas.
#
# Ele herda do InMemoryTaskManager para:
# - Receber uma tarefa do usuário
# - Extrair a pergunta (como "Que horas são?")
# - Pedir ao agente (OpenAI) para responder
# - Salvar e retornar a resposta do agente
# =============================================================================


# -----------------------------------------------------------------------------
# 📚 Importações
# -----------------------------------------------------------------------------

import logging  # Módulo de logging nativo do Python

# 🔁 Importa o gerenciador de tarefas base compartilhado do servidor
from server.task_manager import InMemoryTaskManager

# 🤖 Importa o agente real que estamos usando (o TellTimeAgent adaptado para OpenAI)
from .agent import TellTimeAgent

# 📦 Importa os modelos de dados usados para estruturar e retornar tarefas
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, Task, TextPart, TaskStatus, TaskState


# -----------------------------------------------------------------------------
# 🪵 Configuração do Logger
# -----------------------------------------------------------------------------
# Isso nos permite imprimir logs formatados como:
# INFO:agentes.tell_time_agent.task_manager:Processando nova tarefa: 12345
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# AgentTaskManager (usaremos 'TellTimeTaskManager' para clareza)
# -----------------------------------------------------------------------------

class TellTimeTaskManager(InMemoryTaskManager):
    """
    🧠 Esta classe conecta o agente (OpenAI) ao sistema de tarefas.

    - Ela "herda" toda a lógica do InMemoryTaskManager
    - Ela sobrescreve a parte que lida com uma nova tarefa (on_send_task)
    - Ela usa o agente (OpenAI) para gerar uma resposta
    """

    def __init__(self, agent: TellTimeAgent):
        super().__init__()    # Chama o construtor da classe pai
        self.agent = agent    # Armazena o agente (TellTimeAgent) como uma propriedade

    # -------------------------------------------------------------------------
    # 🔍 Extrai a consulta (query) do usuário da tarefa recebida
    # -------------------------------------------------------------------------
    def _get_user_query(self, request: SendTaskRequest) -> str:
        """
        Obtém a entrada de texto do usuário do objeto da requisição.

        Exemplo: Se o usuário diz "que horas são?", nós extraímos essa string.

        Args:
            request: Um objeto SendTaskRequest

        Returns:
            str: O texto real que o usuário perguntou
        """
        return request.params.message.parts[0].text

    # -------------------------------------------------------------------------
    # 🧠 Lógica principal para lidar e completar uma tarefa
    # -------------------------------------------------------------------------
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Este é o coração do gerenciador de tarefas.

        Ele faz o seguinte:
        1. Salva a tarefa na memória (ou a atualiza)
        2. Pede ao agente (OpenAI) por uma resposta
        3. Formata essa resposta como uma mensagem
        4. Salva a resposta do agente no histórico da tarefa
        5. Retorna a tarefa atualizada para o chamador
        """

        logger.info(f"Processando nova tarefa: {request.params.id}")

        # Passo 1: Salva a tarefa usando o método auxiliar da classe base
        task = await self.upsert_task(request.params)

        # Passo 2: Obtém o que o usuário perguntou
        query = self._get_user_query(request)

        # Passo 3: Pede ao agente (OpenAI) para responder
        # (esta é uma chamada assíncrona 'await' ao nosso agent.invoke)
        result_text = await self.agent.invoke(query, request.params.sessionId)

        # Passo 4: Transforma a resposta do agente em um objeto Message
        agent_message = Message(
            role="agent",                   # O 'role' (papel) é "agent" e não "user"
            parts=[TextPart(text=result_text)]  # O texto de resposta é armazenado dentro de um TextPart
        )

        # Passo 5: Atualiza o estado da tarefa e adiciona a mensagem ao histórico
        async with self.lock:               # Trava o acesso para evitar escritas concorrentes
            task.status = TaskStatus(state=TaskState.COMPLETED)  # Marca a tarefa como concluída
            task.history.append(agent_message)  # Anexa a mensagem do agente ao histórico da tarefa

        # Passo 6: Retorna uma resposta estruturada de volta ao cliente A2A
        return SendTaskResponse(id=request.id, result=task)