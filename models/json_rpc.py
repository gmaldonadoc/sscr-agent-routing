# =============================================================================
# models/json_rpc.py
# =============================================================================
# Propósito:
# Este arquivo define as classes base e estruturas para mensagens JSON-RPC 2.0.
#
# JSON-RPC é um protocolo leve de chamada de procedimento remoto (RPC)
# codificado em JSON.
# Estes modelos são usados para padronizar como os agentes enviam e recebem
# requisições, resultados e erros ao se comunicarem em uma rede
# Agente-para-Agente (A2A).
#
# Isto inclui:
# - JSONRPCMessage: O modelo base para todas as mensagens
# - JSONRPCRequest: Uma chamada de um agente para outro
# - JSONRPCResponse: A resposta a uma requisição (seja resultado ou erro)
# - JSONRPCError: A estrutura de uma resposta de erro
# - InternalError: Um erro padrão predefinido para falhas inesperadas
# =============================================================================

# -----------------------------------------------------------------------------
# Importações
# -----------------------------------------------------------------------------

from typing import Any, Literal         # Para tipos flexíveis e valores literais fixos
from uuid import uuid4                  # Para gerar IDs de requisição únicos
from pydantic import BaseModel, Field   # Para criar modelos de dados robustos e validados


# -----------------------------------------------------------------------------
# JSONRPCMessage (classe base)
# -----------------------------------------------------------------------------
# Todas as mensagens em JSON-RPC compartilham estes campos.
# Esta é a classe pai comum para requisições e respostas.
class JSONRPCMessage(BaseModel):
    # Sempre especifica a versão do protocolo. "2.0" é o único valor válido.
    jsonrpc: Literal["2.0"] = "2.0"

    # O ID da mensagem é usado para parear requisições com respostas.
    # Se não for fornecido, geramos um ID único usando uuid4.
    id: int | str | None = Field(default_factory=lambda: uuid4().hex)


# -----------------------------------------------------------------------------
# JSONRPCRequest (Requisição JSON-RPC)
# -----------------------------------------------------------------------------
# Uma requisição JSON-RPC para chamar um método em outro agente.
# Isto é o que você envia para realizar uma ação.
class JSONRPCRequest(JSONRPCMessage):
    # O nome do método que você quer chamar (ex: "tasks/send")
    method: str

    # Parâmetros de entrada opcionais para o método (podem ser omitidos se não forem necessários)
    params: dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# JSONRPCError (Erro JSON-RPC)
# -----------------------------------------------------------------------------
# Isto representa um objeto de erro padrão em uma resposta JSON-RPC.
# É usado quando a chamada do método falha devido a um erro.
class JSONRPCError(BaseModel):
    # Código de erro numérico. Use códigos padrão se possível (ex: -32603 para erro interno).
    code: int

    # Mensagem legível descrevendo o erro
    message: str

    # Informações adicionais opcionais, como um stack trace ou informações internas de debug
    data: Any | None = None


# -----------------------------------------------------------------------------
# JSONRPCResponse (Resposta JSON-RPC)
# -----------------------------------------------------------------------------
# Uma resposta JSON-RPC que contém ou um resultado ou um erro.
# Apenas um entre `result` ou `error` deve ser não-nulo.
class JSONRPCResponse(JSONRPCMessage):
    # O resultado bem-sucedido de uma chamada de método
    result: Any | None = None

    # O objeto de erro se o método falhou
    error: JSONRPCError | None = None


# -----------------------------------------------------------------------------
# InternalError (subclasse de JSONRPCError)
# -----------------------------------------------------------------------------
# Um erro predefinido para quando o agente encontra uma exceção inesperada.
# Isto segue o código de erro padrão do JSON-RPC para erros internos (-32603).
class InternalError(JSONRPCError):
    # Código de erro fixo para erros internos
    code: int = -32603

    # Mensagem de erro padrão descrevendo o tipo de erro
    message: str = "Internal error"

    # Detalhes de debug opcionais (ex: traceback ou informações de contexto)
    data: Any | None = None