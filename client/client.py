# =============================================================================
# client/client.py
# =============================================================================
# Propósito:
# Este arquivo define um cliente Python assíncrono e reutilizável para interagir
# com um servidor Agent2Agent (A2A).
#
# Ele suporta:
# - Enviar tarefas e receber respostas
# - Obter o status ou histórico da tarefa
# - (Streaming e cancelamento não são suportados nesta versão simplificada)
# =============================================================================

# -----------------------------------------------------------------------------
# Importações
# -----------------------------------------------------------------------------

import json
from uuid import uuid4                    # Usado para codificar/decodificar dados JSON
import httpx                                # Cliente HTTP assíncrono para fazer requisições web
from httpx_sse import connect_sse           # Extensão de cliente SSE para httpx (não usada atualmente)
from typing import Any                      # Dicas de tipo (type hints) para entrada/saída flexível

# Importa tipos de requisição suportados
from models.request import SendTaskRequest, GetTaskRequest  # Removeu CancelTaskRequest

# Formato base de requisição para JSON-RPC 2.0
from models.json_rpc import JSONRPCRequest

# Modelos para resultados de tarefas e identidade do agente
from models.task import Task, TaskSendParams
from models.agent import AgentCard


# -----------------------------------------------------------------------------
# Classes de Erro Customizadas
# -----------------------------------------------------------------------------

class A2AClientHTTPError(Exception):
    """Lançado quando uma requisição HTTP falha (ex: resposta ruim do servidor)"""
    pass

class A2AClientJSONError(Exception):
    """Lançado quando a resposta não é um JSON válido"""
    pass


# -----------------------------------------------------------------------------
# A2AClient: Interface principal para falar com um agente A2A
# -----------------------------------------------------------------------------

class A2AClient:
    def __init__(self, agent_card: AgentCard = None, url: str = None):
        """
        Inicializa o cliente usando ou um 'agent card' ou uma URL direta.
        Um dos dois deve ser fornecido.
        """
        if agent_card:
            self.url = agent_card.url
        elif url:
            self.url = url
        else:
            raise ValueError("É necessário fornecer 'agent_card' ou 'url'")


    # -------------------------------------------------------------------------
    # send_task: Envia uma nova tarefa para o agente
    # -------------------------------------------------------------------------
    async def send_task(self, payload: dict[str, Any]) -> Task:
        """
        Cria e envia uma 'SendTaskRequest' (JSON-RPC) para o agente.
        Recebe um payload (dicionário) do AgentConnector.
        """

        # Constrói a requisição 'tasks/send' completa
        request = SendTaskRequest(
            id=uuid4().hex,  # ID da requisição JSON-RPC
            # ✅ "Embrulha" (wrap) o payload do dict no modelo Pydantic correto
            params=TaskSendParams(**payload) 
        )

        print("\n📤 Enviando requisição JSON-RPC:")
        print(json.dumps(request.model_dump(), indent=2))

        # Envia a requisição e obtém a resposta (dict)
        response = await self._send_request(request)
        
        # ✅ Extrai apenas o campo 'result' e o valida no modelo Task
        return Task(**response["result"])



    # -------------------------------------------------------------------------
    # get_task: Recupera o status ou histórico de uma tarefa enviada anteriormente
    # -------------------------------------------------------------------------
    async def get_task(self, payload: dict[str, Any]) -> Task:
        """ Cria e envia uma 'GetTaskRequest' (JSON-RPC) para o agente. """
        request = GetTaskRequest(params=payload)
        response = await self._send_request(request)
        return Task(**response["result"])



    # -------------------------------------------------------------------------
    # _send_request: Método auxiliar interno para enviar uma requisição JSON-RPC
    # -------------------------------------------------------------------------
    async def _send_request(self, request: JSONRPCRequest) -> dict[str, Any]:
        """
        Método central que usa httpx para enviar a requisição POST
        e lida com erros de rede e JSON.
        """
        # 'httpx.AsyncClient' é usado em um 'with' para garantir
        # que a conexão seja fechada corretamente.
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.url,
                    # Converte o modelo Pydantic em JSON
                    json=request.model_dump(), 
                    timeout=60
                )
                
                # Lança um erro se o código de status for 4xx/5xx
                response.raise_for_status()   
                
                # Retorna a resposta analisada (parse) como um dict
                return response.json()        

            except httpx.HTTPStatusError as e:
                # Captura erros de HTTP (404, 500, etc.)
                raise A2AClientHTTPError(e.response.status_code, str(e)) from e

            except json.JSONDecodeError as e:
                # Captura erros se a resposta do servidor não for um JSON válido
                raise A2AClientJSONError(str(e)) from e