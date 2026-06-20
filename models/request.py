# =============================================================================
# models/request.py
# =============================================================================
# Propósito:
# Este módulo define modelos de requisição estruturados usados no protocolo A2A (Agent2Agent).
#
# Esses modelos representam os diferentes tipos de requisições que um agente pode
# enviar ou receber, como enviar uma tarefa ou recuperar uma tarefa. Cada requisição
# adere ao formato JSON-RPC 2.0.
#
# Ele também inclui uma "união discriminada" (discriminated union) chamada `A2ARequest`,
# que automaticamente identifica e analisa (parse) a requisição com base no
# campo `method`.
#
# Modelos Incluídos:
# - SendTaskRequest
# - GetTaskRequest
# - A2ARequest (união discriminada)
# - SendTaskResponse
# - GetTaskResponse
#
# Nota: CancelTaskRequest será adicionada em uma versão futura se o
# suporte a cancelamento for implementado.
# =============================================================================

# -----------------------------------------------------------------------------
# Importações
# -----------------------------------------------------------------------------

from typing import Annotated, Union, Literal    # Para anotações de tipo e lógica de discriminador
from pydantic import Field                    # Configurações de campo para modelos Pydantic
from pydantic.type_adapter import TypeAdapter   # Para análise (parsing) de união discriminada em tempo de execução

# Modelos base para requisições e respostas JSON-RPC
from models.json_rpc import JSONRPCRequest, JSONRPCResponse

# Modelos de parâmetros e retorno relacionados a tarefas
# (Estes virão do models/task.py)
from models.task import Task, TaskSendParams
from models.task import TaskQueryParams


# -----------------------------------------------------------------------------
# SendTaskRequest: Usado para enviar uma nova tarefa a um agente
# -----------------------------------------------------------------------------

class SendTaskRequest(JSONRPCRequest):
    """
    Define uma requisição 'tasks/send'. Herda 'jsonrpc' e 'id' de JSONRPCRequest.
    """
    method: Literal["tasks/send"] = "tasks/send"   # String exata do método obrigatória
    params: TaskSendParams                         # Parâmetros de criação da tarefa


# -----------------------------------------------------------------------------
# GetTaskRequest: Usado para recuperar o status ou histórico de uma tarefa
# -----------------------------------------------------------------------------

class GetTaskRequest(JSONRPCRequest):
    """
    Define uma requisição 'tasks/get'. Herda 'jsonrpc' e 'id' de JSONRPCRequest.
    """
    method: Literal["tasks/get"] = "tasks/get"     # String exata do método obrigatória
    params: TaskQueryParams                        # ID da tarefa e limite de histórico opcional


# -----------------------------------------------------------------------------
# A2ARequest: União discriminada dos tipos de requisição suportados
# -----------------------------------------------------------------------------
# Isso permite a análise (parsing) automática dos tipos de requisição com base no campo `method`.
# O 'server.py' usará isso para validar o JSON recebido.

A2ARequest = TypeAdapter(
    Annotated[
        Union[
            SendTaskRequest,
            GetTaskRequest,
            # CancelTaskRequest pode ser adicionado aqui no futuro se implementado
        ],
        # 'Field(discriminator="method")' diz ao Pydantic:
        # "Olhe para o campo 'method' no JSON de entrada para decidir
        # qual modelo (SendTaskRequest ou GetTaskRequest) usar para validação."
        Field(discriminator="method")
    ]
)


# -----------------------------------------------------------------------------
# SendTaskResponse: Modelo de resposta para uma requisição "tasks/send" bem-sucedida
# -----------------------------------------------------------------------------

class SendTaskResponse(JSONRPCResponse):
    """
    Define a resposta para 'tasks/send'. Herda 'jsonrpc' e 'id' de JSONRPCResponse.
    """
    # A tarefa retornada pelo agente (ou None se não houver resultado)
    result: Task | None = None


# -----------------------------------------------------------------------------
# GetTaskResponse: Modelo de resposta para uma requisição "tasks/get"
# -----------------------------------------------------------------------------

class GetTaskResponse(JSONRPCResponse):
    """
    Define a resposta para 'tasks/get'. Herda 'jsonrpc' e 'id' de JSONRPCResponse.
    """
    # A tarefa solicitada, ou None se não for encontrada
    result: Task | None = None