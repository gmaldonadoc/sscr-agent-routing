# =============================================================================
# agents/host_agent/orchestrator.py
# =============================================================================
# 🎯 Propósito:
# Define o OrchestratorAgent, que:
#   1) Descobre e chama outros agentes A2A (via DiscoveryClient & AgentConnector)
#   2) Descobre e carrega ferramentas MCP (via MCPConnector)
#   3) Expõe cada ação A2A e cada ferramenta MCP como sua própria "ferramenta"
#      chamável para um LLM da OpenAI.
#   4) Roteia a consulta do usuário para a ferramenta correta usando o LLM.
#   - Suporta modo 'Simples' (lista todos os agentes no prompt).
#   - Suporta modo 'RAG' (filtra agentes via ChromaDB por similaridade).
# Também define OrchestratorTaskManager para servir este agente sobre JSON-RPC.
# =============================================================================

import logging
import json
import random
from dotenv import load_dotenv
from openai import OpenAI

from server.task_manager import InMemoryTaskManager
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, TaskStatus, TaskState, TextPart

from utilities.a2a.agent_connect import AgentConnector
from utilities.mcp.mcp_connect import MCPConnector
from utilities.routing.structural_router import StructuralAgentRouter

from models.agent import AgentCard
# -----------------------------------------------------------------------------
# Configuração de Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OrchestratorAgent:
    """
    🤖 OrchestratorAgent (Versão OpenAI):
     - Usa um LLM da OpenAI (ex: gpt-4o) como o cérebro de roteamento.
     - Conecta-se a Agentes A2A e ferramentas MCP.
     - Expõe todos eles como "Ferramentas" (Tools) para o LLM.
     - Roteia as consultas do usuário invocando a ferramenta correta.
    """

    def __init__(self, agent_cards: list[AgentCard], structural_mode: bool = True, pool_size: int = 5):
        self.structural_mode = structural_mode
        self.pool_size = pool_size

        self.client = OpenAI()
        self.model = "gpt-4.1-nano"

        self.connectors = {}
        self.agent_descriptions = {}
        self.agent_cards_snapshot = []
        self.last_llm_pool_prompt = None

        if agent_cards:
            logger.info(f"⚡ Preparando {len(agent_cards)} agentes...")
            for card in agent_cards:
                self.connectors[card.name] = AgentConnector(card.name, card.url)
                self.agent_descriptions[card.name] = card.description
                if hasattr(card, "model_dump"):
                    self.agent_cards_snapshot.append(card.model_dump())
                else:
                    self.agent_cards_snapshot.append(card.dict())

        self.structural_router = None
        if self.structural_mode:
            logger.info("🧭 MODO STRUCTURAL ROUTER ATIVO")
            self.structural_router = StructuralAgentRouter(
                agent_cards=agent_cards,
                pool_size=self.pool_size
            )

        self.mcp = MCPConnector()
        mcp_tools = self.mcp.get_tools()
        self.mcp_tool_map = {t.name: t.run for t in mcp_tools}

        self.tool_definitions = []
        self.tool_implementations = {}

        for tool in mcp_tools:
            self.tool_definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
            self.tool_implementations[tool.name] = tool.run

        self.tool_definitions.extend([
            {
                "type": "function",
                "function": {
                    "name": "list_agents",
                    "description": "Lista nomes dos agentes A2A disponíveis."
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delegate_task",
                    "description": "Delega uma tarefa para um agente A2A específico.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {"type": "string", "description": "Nome exato do agente."},
                            "message": {"type": "string", "description": "A tarefa a ser enviada."}
                        },
                        "required": ["agent_name", "message"]
                    }
                }
            }
        ])

        self.tool_implementations["list_agents"] = self._list_agents
        self.tool_implementations["delegate_task"] = self._delegate_task

        self.sessions = {}

    def _get_structural_agents_text(self, user_query: str) -> str:
        if not self.connectors:
            return "Nenhum agente disponível."

        if not self.structural_router:
            logger.warning("StructuralRouter não pronto. Usando fallback com lista completa.")
            return self._get_all_agents_text()

        try:
            trace = self.structural_router.route_with_trace(user_query)
            self.last_routing_trace = trace

            pool = trace["compact_pool"]

            if not pool:
                return "  - Nenhum agente relevante encontrado."

            lines = []
            for item in pool:
                lines.append(
                    f"  - '{item['agent_name']}': {item['description']} "
                    f"(score={item['score']}, route={item['tree_keys']})"
                )

            text = "\n".join(lines)
            logger.info(f"[STRUCTURAL ROUTER] Agentes retornados:\n{text}")
            return text

        except Exception as e:
            logger.error(f"Erro no StructuralRouter: {e}")
            return self._get_all_agents_text()
        

    def _get_all_agents_text(self) -> str:
        """ Retorna a lista completa de agentes (Modo Simples). """
        text = ""
        
        # --- FAIRNESS FIX (BASE ONLY) ---
        # Como não existe "ordem de relevância" aqui, embaralhamos
        # para que o TellTimeAgent não ganhe vantagem injusta por estar na linha 1.
        all_agents = list(self.agent_descriptions.items())
        random.shuffle(all_agents) 
        # --------------------------------

        for name, desc in all_agents:
            text += f"  - '{name}': {desc}\n"
            
        return text if text else "  - Nenhum agente disponível."


 
    # --- Prompt Dinâmico 
    def _system_prompt(self, user_query: str = "") -> dict:
        mcp_tool_names = ", ".join(self.mcp_tool_map.keys()) or "Nenhuma"

        if self.structural_mode and user_query:
            agents_list_text = self._get_structural_agents_text(user_query)
            mode_note = "(Filtrado via STRUCTURAL ROUTER)"
        else:
            agents_list_text = self._get_all_agents_text()
            mode_note = "(Lista Completa)"

        content = (
            "Você é um orquestrador de roteamento de IA. Analise o pedido e use ferramentas quando necessário.\n"
            "Na primeira etapa, escolha uma ferramenta adequada para atender ao pedido.\n"
            "Depois que uma ferramenta retornar resultado, gere uma resposta final em texto para o usuário.\n"
            "Nunca responda apenas com a palavra `tool_calls`.\n\n"

            f"**1. AGENTES A2A DISPONÍVEIS {mode_note}:**\n"
            "Use `delegate_task(agent_name, message)` escolhendo um agente do pool compacto abaixo:\n"
            f"{agents_list_text}\n"

            "**2. FERRAMENTAS LOCAIS (MCP):**\n"
            f"  - Disponíveis: [{mcp_tool_names}]\n"
            "Se o usuário pedir para escrever, salvar ou gravar em arquivo, você DEVE chamar `run_shell_command`.\n"
            "Não finalize a resposta enquanto o arquivo não tiver sido gravado.\n"
            "Depois de `run_shell_command` retornar OK, então responda ao usuário.\n\n"

            "**PEDIDOS COMPLEXOS:**\n"
            "Se o usuário pedir várias etapas, chame as ferramentas em sequência "
            "(ex: delegue primeiro, receba o resultado, depois use MCP)."
        )
        self.last_llm_pool_prompt = content
        return {"role": "system", "content": content}

    # # --- Implementações de Ferramentas ---
    def _list_agents(self) -> list[str]:
        return list(self.connectors.keys())

    async def _delegate_task(self, agent_name: str, message: str, session_id: str) -> str:
        if agent_name not in self.connectors:
            return f"Erro: Agente '{agent_name}' não encontrado."
        try:
            task = await self.connectors[agent_name].send_task(message, session_id)
            if task.history and len(task.history) > 1:
                return task.history[-1].parts[0].text
            return "(Tarefa concluída sem resposta de texto)"
        except Exception as e:
            return f"Erro na delegação: {e}"

    # --- Invoke Principal ---
    async def invoke(self, query: str, session_id: str) -> tuple[str, list]:
        """
        Ponto de entrada principal: lida com uma consulta do usuário.
        
        Etapas (Ciclo de Chamada de Ferramenta da OpenAI):
        1.  Obtém/Cria o histórico de mensagens para a sessão.
        2.  Adiciona a consulta do usuário ao histórico.
        3.  Chama a API da OpenAI com o histórico e a lista de ferramentas.
        4.  Verifica se o LLM pediu para chamar uma ferramenta.
        5.  Se SIM:
            a. Executa a(s) ferramenta(s) (ex: _delegate_task, _list_agents).
            b. Adiciona os resultados da ferramenta ao histórico.
            c. Chama a API da OpenAI *novamente* com o histórico atualizado.
            d. Retorna a resposta final de texto do LLM.
        6.  Se NÃO (resposta de texto direta):
            a. Retorna a resposta de texto do LLM.
        7.  Atualiza o System Prompt dinamicamente com base na query (para RAG).
        """
        
        # 1. Gera o Prompt de Sistema atualizado (RAG ou Simples)
        current_prompt = self._system_prompt(user_query=query)
        
        # 2. Gerencia a Sessão
        if session_id not in self.sessions: self.sessions[session_id] = [current_prompt]
        else: self.sessions[session_id][0] = current_prompt
        
        messages = self.sessions[session_id]
        messages.append({"role": "user", "content": query})

        tool_usage_history = [] # Rastreamento
        
        try:
            while True:
                tool_choice = "required" if len(messages) == 2 else "auto"

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice=tool_choice,
                    temperature=0.1
                )

                msg = response.choices[0].message

                print("\n====================")
                print("LLM RESPONSE")
                print("====================")
                print(msg)
                print("====================\n")

                messages.append(msg)

                if not msg.tool_calls:
                    return (msg.content or "Concluído."), tool_usage_history

                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")

                    usage = {"tool": name}
                    if name == "delegate_task":
                        usage["agent"] = args.get("agent_name", "Unknown")
                    tool_usage_history.append(usage)

                    if name not in self.tool_implementations:
                        res = "Ferramenta não encontrada."
                    else:
                        try:
                            fn = self.tool_implementations[name]

                            if name == "delegate_task":
                                args["session_id"] = session_id
                                res = await fn(**args)

                            elif name in self.mcp_tool_map:
                                # MCPTool.run espera um dict único
                                res = await fn(args)

                            else:
                                res = fn(**args)

                            print("\n====================")
                            print("TOOL RESULT")
                            print("====================")
                            print(res)
                            print("====================\n")

                        except Exception as e:
                            res = f"Erro: {e}"

                    messages.append({
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "name": name,
                        "content": str(res)
                    })

        except Exception as e:
            return f"Erro: {e}", []

# --- TaskManager (Wrapper) ---
class OrchestratorTaskManager(InMemoryTaskManager):
    """
    Wrapper do TaskManager: expõe o OrchestratorAgent.invoke()
    sobre o endpoint JSON-RPC `tasks/send`.
    
    (Esta classe permanece quase idêntica à versão do ADK,
    pois sua lógica é independente do framework do LLM.)
    """
    def __init__(self, agent: OrchestratorAgent):
        super().__init__()
        self.agent = agent
    
    def _get_user_text(self, r): return r.params.message.parts[0].text
    
    async def on_send_task(self, req: SendTaskRequest) -> SendTaskResponse:
        """
        Lida com chamadas `tasks/send`:
         1) Armazena a mensagem recebida na memória
         2) Invoca o orquestrador para obter uma resposta
         3) Anexa a resposta, marca a tarefa como CONCLUÍDA
         4) Retorna a Tarefa (Task) completa na resposta
        """
        # Armazena ou atualiza o registro da tarefa
        task = await self.upsert_task(req.params)
        
        # Extrai o texto e invoca a lógica de orquestração
        reply_text, tool_usage_history = await self.agent.invoke(
            self._get_user_text(req),
            req.params.sessionId
        )

        task.metadata = task.metadata or {}
        task.metadata["tools_used"] = tool_usage_history
        task.metadata["agent_cards_snapshot"] = getattr(self.agent, "agent_cards_snapshot", [])
        task.metadata["llm_pool_prompt"] = getattr(self.agent, "last_llm_pool_prompt", None)
        routing_trace = getattr(self.agent, "last_routing_trace", None)
        task.metadata["routing_trace"] = routing_trace
        

        if routing_trace:
            query_signals = set(
                routing_trace.get("request_profile", {}).get("query_signals", [])
            )

            for item in tool_usage_history:
                if item.get("tool") == "delegate_task":
                    agent_name = item.get("agent")
                    if agent_name and self.agent.structural_router:
                        self.agent.structural_router.signal_graph.memory.record_result(
                            agent_name=agent_name,
                            query_signals=query_signals,
                            success=True
                        )
        #task.metadata["routing_trace"] = getattr(self.agent, "last_routing_trace", None)
        msg = Message(role="agent", parts=[TextPart(text=reply_text)])
        
        # Anexa com segurança a resposta e atualiza o status sob um lock
        async with self.lock:
            task.status = TaskStatus(state=TaskState.COMPLETED)
            task.history.append(msg)

        # Retorna a resposta RPC incluindo a tarefa atualizada
        return SendTaskResponse(id=req.id, result=task)