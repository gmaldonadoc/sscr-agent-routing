# =============================================================================
# utilities/a2a/agent_connect.py
# =============================================================================
# 🎯 Propósito:
# Fornece um wrapper simples (`AgentConnector`) em torno do `A2AClient` para
# enviar tarefas para qualquer agente remoto identificado por uma URL base.
# Isso desacopla o Orquestrador dos detalhes de baixo nível do HTTP
# e da configuração do cliente HTTP.
# =============================================================================

import uuid                             # Biblioteca padrão para gerar IDs únicos
import logging                          # Biblioteca padrão para logging configurável

# Importa nosso A2AClient customizado que lida com requisições de tarefa JSON-RPC
from client.client import A2AClient
# Importa o modelo Task para representar a resposta completa da tarefa
from models.task import Task

# Cria um logger para este módulo usando seu namespace
logger = logging.getLogger(__name__)


class AgentConnector:
    """
    🔗 Conecta-se a um agente A2A remoto e fornece um método uniforme
    para delegar tarefas.

    Atributos:
        name (str): Identificador legível do agente remoto.
        client (A2AClient): Cliente HTTP apontando para a URL do agente.
    """

    def __init__(self, name: str, base_url: str):
        """
        Inicializa o conector para um agente remoto específico.

        Args:
            name (str): Identificador para o agente (ex: "AgenteDeHorario").
            base_url (str): O endpoint HTTP (ex: "http://localhost:10000").
        """
        # Armazena o nome do agente para logging e referência
        self.name = name
        # Instancia um A2AClient vinculado à URL base do agente
        self.client = A2AClient(url=base_url)
        # Loga que o conector está pronto para uso
        logger.info(f"AgentConnector: inicializado para {self.name} em {base_url}")

    async def send_task(self, message: str, session_id: str) -> Task:
        """
        Envia uma tarefa de texto para o agente remoto e retorna a
        Tarefa (Task) completa.

        Args:
            message (str): O que você quer que o agente faça (ex: "Que horas são?").
            session_id (str): Identificador de sessão para agrupar chamadas relacionadas.

        Returns:
            Task: O objeto Task completo (incluindo histórico) do agente remoto.
        """
        # Gera um ID único para esta tarefa usando uuid4, em formato hexadecimal
        task_id = uuid.uuid4().hex
        
        # Constrói a carga útil (payload) JSON-RPC que corresponde ao esquema TaskSendParams
        # (Note que aqui estamos construindo um dicionário puro, não um modelo Pydantic)
        payload = {
            "id": task_id,
            "sessionId": session_id,
            "message": {
                "role": "user",                # Indica que esta mensagem é do usuário
                "parts": [                     # Envolve o texto em uma lista de partes
                    {"type": "text", "text": message}
                ]
            }
        }

        # Usa o A2AClient para enviar a tarefa assincronamente e aguarda a resposta
        task_result = await self.client.send_task(payload)
        
        # Loga o recebimento da tarefa completa para debug/rastreamento
        logger.info(f"AgentConnector: resposta recebida de {self.name} para tarefa {task_id}")
        
        # Retorna o modelo Pydantic Task para processamento futuro pelo orquestrador
        return task_result