# =============================================================================
# server.py
# =============================================================================
# 📌 Propósito:
# Este arquivo define um servidor A2A (Agente-para-Agente) muito simples.
# Ele suporta:
# - Receber requisições de tarefas via POST ("/")
# - Permitir que clientes descubram os detalhes do agente via GET ("/.well-known/agent.json")
# NOTA: Esta versão não suporta streaming ou notificações push.
# =============================================================================


# -----------------------------------------------------------------------------
# 🧱 Importações Necessárias
# -----------------------------------------------------------------------------

# 🌐 Starlette é um framework web leve para construir aplicações ASGI
from starlette.applications import Starlette     # Para criar nossa aplicação web
from starlette.responses import JSONResponse     # Para enviar respostas como JSON
from starlette.requests import Request           # Representa requisições HTTP recebidas

# 📦 Importando nossos modelos e lógica customizada
from models.agent import AgentCard               # Descreve a identidade e habilidades do agente
from models.request import A2ARequest, SendTaskRequest # Modelos de requisição para tarefas
from models.json_rpc import JSONRPCResponse, InternalError # Utilitários JSON-RPC para mensagens estruturadas
from server import task_manager                  # Nossa lógica real de manipulação de tarefas (o agente)

# 🛠️ Utilitários gerais
import json                                      # Usado para imprimir os dados da requisição (para debug)
import logging                                   # Usado para registrar logs de erros e informações
logger = logging.getLogger(__name__)             # Configura um logger para este arquivo

# 🕒 Importação de datetime para serialização
from datetime import datetime

# 📦 Encoder para ajudar a converter dados complexos como datetime em JSON
from fastapi.encoders import jsonable_encoder


# -----------------------------------------------------------------------------
# 🔧 Serializador para datetime (Opcional, pois usamos jsonable_encoder)
# -----------------------------------------------------------------------------
def json_serializer(obj):
    """
    Esta função pode converter objetos datetime do Python para strings ISO.
    Se você tentar serializar um tipo que ela não conhece, ela lançará um erro.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Tipo {type(obj)} não serializável")


# -----------------------------------------------------------------------------
# 🚀 Classe A2AServer: A Lógica Central do Servidor
# -----------------------------------------------------------------------------
class A2AServer:
    def __init__(self, host="0.0.0.0", port=5000, agent_card: AgentCard = None, task_manager: task_manager = None):
        """
        🔧 Construtor para nosso A2AServer

        Args:
            host (str): Endereço IP ao qual o servidor será vinculado (padrão é todas as interfaces)
            port (int): Número da porta que o servidor ouvirá (padrão é 5000)
            agent_card (AgentCard): Metadados que descrevem nosso agente (nome, habilidades, capacidades)
            task_manager (TaskManager): A lógica para manipular a tarefa (ex: nosso OrchestratorTaskManager)
        """
        self.host = host
        self.port = port
        self.agent_card = agent_card
        self.task_manager = task_manager

        # 🌐 Inicialização da aplicação Starlette
        self.app = Starlette()

        # 📥 Registra uma rota para lidar com requisições de tarefas (JSON-RPC POST)
        self.app.add_route("/", self._handle_request, methods=["POST"])

        # 🔎 Registra uma rota para descoberta de agente (metadados como JSON)
        self.app.add_route("/.well-known/agent.json", self._get_agent_card, methods=["GET"])

    # -----------------------------------------------------------------------------
    # ▶️ start(): Inicia o servidor web usando uvicorn
    # -----------------------------------------------------------------------------
    def start(self):
        """
        Inicia o servidor A2A usando uvicorn (servidor web ASGI).
        Esta função irá bloquear e executar o servidor indefinidamente.
        """
        if not self.agent_card or not self.task_manager:
            raise ValueError("Agent card e task manager são obrigatórios")

        # Importa uvicorn dinamicamente para que seja carregado apenas quando necessário
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)

    # -----------------------------------------------------------------------------
    # 🔎 _get_agent_card(): Retorna os metadados do agente (requisição GET)
    # -----------------------------------------------------------------------------
    def _get_agent_card(self, request: Request) -> JSONResponse:
        """
        Endpoint para descoberta de agente (GET /.well-known/agent.json)

        Returns:
            JSONResponse: Metadados do agente como um dicionário
        """
        # .model_dump() converte o objeto Pydantic (agent_card) em um dicionário
        return JSONResponse(self.agent_card.model_dump(exclude_none=True))

    # -----------------------------------------------------------------------------
    # 📥 _handle_request(): Lida com requisições POST recebidas para tarefas
    # -----------------------------------------------------------------------------
    async def _handle_request(self, request: Request):
        """
        Este método lida com requisições de tarefas enviadas para a rota raiz ("/").

        - Analisa (parse) o JSON recebido
        - Valida a mensagem JSON-RPC
        - Para tipos de tarefas suportados, delega ao task_manager
        - Retorna uma resposta ou erro
        """
        try:
            # Passo 1: Analisar o corpo JSON recebido
            body = await request.json()
            print("\n🔍 JSON Recebido:", json.dumps(body, indent=2)) # Loga a entrada para visibilidade

            # Passo 2: Analisar e validar a requisição usando Pydantic
            # A2ARequest (provavelmente um 'discriminated union') descobre o tipo
            json_rpc = A2ARequest.validate_python(body)

            # Passo 3: Se for uma requisição 'send-task', chama o task_manager para lidar com ela
            if isinstance(json_rpc, SendTaskRequest):
                # O task_manager (nosso OrchestratorTaskManager) fará o trabalho
                result = await self.task_manager.on_send_task(json_rpc)
            else:
                raise ValueError(f"Método A2A não suportado: {type(json_rpc)}")

            # Passo 4: Converte o resultado em uma resposta JSON adequada
            return self._create_response(result)

        except Exception as e:
            logger.error(f"Exceção: {e}")
            # Retorna uma resposta de erro compatível com JSON-RPC se algo falhar
            return JSONResponse(
                JSONRPCResponse(id=None, error=InternalError(message=str(e))).model_dump(),
                status_code=400
            )

    # -----------------------------------------------------------------------------
    # 🧾 _create_response(): Converte o objeto de resultado em uma JSONResponse
    # -----------------------------------------------------------------------------
    def _create_response(self, result):
        """
        Converte um objeto JSONRPCResponse em uma resposta HTTP JSON.

        Args:
            result: O objeto de resposta (deve ser um JSONRPCResponse)

        Returns:
            JSONResponse: Resposta HTTP compatível com Starlette com corpo JSON
        """
        if isinstance(result, JSONRPCResponse):
            # jsonable_encoder lida automaticamente com tipos como datetime e UUID
            # .model_dump() converte o Pydantic em um dict
            return JSONResponse(content=jsonable_encoder(result.model_dump(exclude_none=True)))
        else:
            raise ValueError("Tipo de resposta inválido")