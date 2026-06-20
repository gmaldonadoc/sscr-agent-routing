# =============================================================================
# agentes/greeting_agent/task_manager.py
# =============================================================================
# 🎯 Propósito:
# Conecta a classe GreetingAgent ao protocolo Agent-to-Agent (A2A) ao
# lidar com requisições JSON-RPC "tasks/send" recebidas. Ele:
# 1. Recebe um modelo SendTaskRequest
# 2. Armazena a mensagem do usuário na memória
# 3. Chama GreetingAgent.invoke() para gerar assincronamente uma saudação
# 4. Atualiza o status da tarefa e anexa a saudação ao histórico da tarefa
# 5. Retorna um SendTaskResponse contendo a Tarefa (Task) concluída
# =============================================================================

# -----------------------------------------------------------------------------
# 📦 Importações
# -----------------------------------------------------------------------------
import logging                      # Módulo de logging nativo do Python

# InMemoryTaskManager fornece armazenamento em memória e lock para tarefas
from server.task_manager import InMemoryTaskManager

# Modelos de dados para lidar com requisições/respostas A2A JSON-RPC e estruturas de tarefas
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, TaskStatus, TaskState, TextPart

# A lógica de negócio principal: GreetingAgent com um método async invoke()
from .agent import GreetingAgent

# -----------------------------------------------------------------------------
# 🪵 Configuração do Logger
# -----------------------------------------------------------------------------
# Cria um logger específico para este módulo usando seu __name__
logger = logging.getLogger(__name__)


class GreetingTaskManager(InMemoryTaskManager):
    """
    🧩 TaskManager para o GreetingAgent:

    - Herda armazenamento, upsert_task e lock do InMemoryTaskManager
    - Sobrescreve on_send_task() para:
      * salvar a mensagem recebida
      * chamar o GreetingAgent.invoke() para criar uma saudação
      * atualizar o status e histórico da tarefa
      * "embrulhar" (wrap) e retornar o resultado como SendTaskResponse

    Nota:
    - GreetingAgent.invoke() é assíncrono, mas on_send_task()
      em si também é definido como async, então usamos 'await' nas chamadas internas.
    """
    def __init__(self, agent: GreetingAgent):
        """
        Inicializa o TaskManager com uma instância do GreetingAgent.

        Args:
            agent (GreetingAgent): O manipulador da lógica principal que sabe como
                                   produzir uma saudação.
        """
        # Chama o construtor pai para configurar self.tasks e self.lock
        super().__init__()
        # Armazena uma referência ao nosso GreetingAgent para uso posterior
        self.agent = agent

    def _get_user_text(self, request: SendTaskRequest) -> str:
        """
        Extrai o texto bruto do usuário da SendTaskRequest recebida.

        Args:
            request (SendTaskRequest): A requisição JSON-RPC recebida
                                       contendo um objeto TaskSendParams.

        Returns:
            str: O conteúdo de texto que o usuário enviou (primeiro TextPart).
        """
        # A request.params.message.parts é uma lista de objetos TextPart.
        # Pegamos o atributo .text do primeiro elemento.
        return request.params.message.parts[0].text

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Lida com uma nova tarefa de saudação:

        1. Armazena a mensagem do usuário recebida na memória (ou atualiza tarefa existente)
        2. Extrai o texto do usuário para processamento
        3. Chama GreetingAgent.invoke() para gerar a saudação
        4. "Embrulha" (wrap) essa string de saudação em uma Message/TextPart
        5. Atualiza o status da Tarefa (Task) para COMPLETED e anexa a resposta
        6. Retorna um SendTaskResponse contendo a Tarefa (Task) atualizada

        Args:
            request (SendTaskRequest): A requisição JSON-RPC com TaskSendParams

        Returns:
            SendTaskResponse: Uma resposta JSON-RPC com a Tarefa (Task) concluída
        """
        # Loga o recebimento de uma nova tarefa junto com seu ID
        logger.info(f"GreetingTaskManager recebeu tarefa {request.params.id}")

        # Passo 1: Salva ou atualiza a tarefa na memória.
        # upsert_task() criará uma nova Tarefa (Task) se ela não existir,
        # ou anexará a mensagem do usuário recebida ao histórico existente.
        task = await self.upsert_task(request.params)

        # Passo 2: Extrai o texto real que o usuário enviou
        user_text = self._get_user_text(request)

        # Passo 3: Chama o GreetingAgent para gerar um texto de saudação.
        # Como GreetingAgent.invoke() é uma função async,
        # usamos 'await' para obter a string retornada.
        greeting_text = await self.agent.invoke(
            user_text,
            request.params.sessionId
        )

        # Passo 4: "Embrulha" (wrap) a string de saudação em um TextPart, e depois em uma Message
        reply_message = Message(
            role="agent",                   # Marca esta como uma resposta do "agente"
            parts=[TextPart(text=greeting_text)]  # O texto de resposta do agente
        )

        # Passo 5: Atualiza o status da tarefa para COMPLETED e anexa nossa resposta
        # Usa o 'lock' para evitar condições de corrida com outras corrotinas.
        async with self.lock:
            # Marca a tarefa como concluída
            task.status = TaskStatus(state=TaskState.COMPLETED)
            # Adiciona a resposta do agente ao histórico da tarefa
            task.history.append(reply_message)

        # Passo 6: Retorna um SendTaskResponse, contendo o ID JSON-RPC
        # (espelhando o request.id) e o modelo Task atualizado.
        return SendTaskResponse(id=request.id, result=task)