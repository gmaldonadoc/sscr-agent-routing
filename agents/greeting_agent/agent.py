# =============================================================================
# agentes/greeting_agent/agent.py
# =============================================================================
# 🎯 Propósito:
#   Um agente "orquestrador" composto que:
#     1) Descobre todos os agentes A2A registrados via DiscoveryClient
#     2) Invoca o TellTimeAgent para buscar a hora atual
#     3) Gera uma saudação poética de 2–3 linhas referenciando essa hora
# =============================================================================

import logging
import json
import asyncio
from dotenv import load_dotenv
from openai import OpenAI  # Importa o cliente OpenAI

load_dotenv()  # Lê o .env (para OPENAI_API_KEY)

# Nossas utilidades para descoberta e conexão
from utilities.a2a.agent_discovery import DiscoveryClient
from utilities.a2a.agent_connect import AgentConnector

# Cria um logger no nível do módulo
logger = logging.getLogger(__name__)


class GreetingAgent:
    """
    🧠 "Meta-agente" Orquestrador (Versão OpenAI) que:
      - Usa a API de "Tool Calling" da OpenAI.
      - Fornece duas ferramentas: list_agents() e call_agent(...)
      - Ao receber um pedido de "saudação":
          1) O LLM chama list_agents()
          2) O LLM chama call_agent("TellTimeAgent", ...)
          3) O LLM cria uma saudação poética com o resultado
    """

    def __init__(self):
        """
        🏗️ Construtor: inicializa o cliente OpenAI, o discovery e as ferramentas.
        """
        # 1. Inicializa o cliente OpenAI (lê a chave do .env)
        self.client = OpenAI()
        self.model = "gpt-4.1-nano"  # 

        # 2. Um cliente auxiliar para descobrir quais agentes estão registrados
        self.discovery = DiscoveryClient()

        # 3. Cache para conectores criados, para que possamos reusá-los
        self.connectors: dict[str, AgentConnector] = {}

        # 4. Armazenamento de sessão (substitui o SessionService do ADK)
        # Mapeia session_id -> list[messages]
        self.sessions = {}

        # 5. Define as ferramentas que a OpenAI pode chamar
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "list_agents",
                    "description": "Busca metadados de todos os agentes A2A disponíveis.",
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "call_agent",
                    "description": "Envia uma mensagem para um agente A2A específico pelo nome e obtém uma resposta.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {
                                "type": "string",
                                "description": "O nome do agente para o qual ligar (ex: 'TellTimeAgent')."
                            },
                            "message": {
                                "type": "string",
                                "description": "A mensagem/pergunta a ser enviada ao agente."
                            }
                        },
                        "required": ["agent_name", "message"]
                    }
                }
            }
        ]

        # 6. Mapeia os nomes das ferramentas para as funções Python reais
        self.tool_implementations = {
            "list_agents": self._list_agents,
            "call_agent": self._call_agent,
        }

    def _get_system_prompt(self) -> dict:
        """ Define a instrução de sistema (prompt) para o LLM. """
        system_instr = (
            "Você é um poeta orquestrador. Você tem duas ferramentas:\n"
            "1) `list_agents()`: Retorna a lista e descrição de todos os agentes disponíveis.\n"
            "2) `call_agent(agent_name, message)`: Envia uma mensagem para um agente.\n\n"
            "**SEU OBJETIVO:**\n"
            "Quando solicitado a saudar ('greet me'), você deve:\n"
            "1. Chamar `list_agents()` para ver quem está disponível.\n"
            "2. **Analisar as descrições** para encontrar um agente capaz de **informar a hora atual**.\n"
            "3. Chamar esse agente perguntando: 'Que horas são?'.\n"
            "4. Com a resposta, criar uma saudação poética de 2-3 linhas.\n"
            "5. Se não encontrar nenhum agente de hora, invente uma hora mágica para o poema."
        )
        return {"role": "system", "content": system_instr}

    # --- Implementações das Ferramentas ---

    async def _list_agents(self) -> list[dict]:
        """
        [Ferramenta] Busca todos os AgentCard, retorna como lista de dicts.
        """
        logger.info("[GreetingAgent Tool] Executando _list_agents()")
        cards = await self.discovery.list_agent_cards()
        return [card.model_dump(exclude_none=True) for card in cards]

    async def _call_agent(self, agent_name: str, message: str) -> str:
        """
        [Ferramenta] Encontra um agente e envia uma tarefa para ele.
        """
        logger.info(f"[GreetingAgent Tool] Executando _call_agent('{agent_name}')...")
        cards = await self.discovery.list_agent_cards()

        # Lógica de correspondência (match)
        matched = next(
            (c for c in cards if c.name.lower() == agent_name.lower()),
            None
        )
        if not matched:
            matched = next(
                (c for c in cards if agent_name.lower() in c.name.lower()),
                None
            )
        if not matched:
            raise ValueError(f"Agente '{agent_name}' não encontrado.")

        # Cria ou reutiliza o conector
        key = matched.name
        if key not in self.connectors:
            self.connectors[key] = AgentConnector(
                name=matched.name,
                base_url=matched.url
            )
        connector = self.connectors[key]

        # Delega a tarefa
        # (Usamos um session_id fixo, pois este agente não mantém estado de longo prazo)
        task = await connector.send_task(message, session_id="greeting_agent_session")

        # Extrai a resposta
        if task.history and task.history[-1].parts:
            return task.history[-1].parts[0].text
        return ""

    # --- Método Invoke (Substitui o 'Runner' do ADK) ---

    async def invoke(self, query: str, session_id: str) -> str:
        """
        🔄 Público: envia uma consulta (query) do usuário através do
        pipeline do LLM OpenAI e lida com o ciclo de chamada de ferramentas.
        """
        
        # 1. Obtém ou cria o histórico de sessão
        if session_id not in self.sessions:
            self.sessions[session_id] = [self._get_system_prompt()]
        messages = self.sessions[session_id]
        
        # 2. Adiciona a consulta do usuário
        messages.append({"role": "user", "content": query})
        
        logger.info(f"[GreetingAgent] Invocando LLM (OpenAI) para session_id {session_id}")

        try:
            # 3. Primeira chamada à API (para decidir se usa ferramentas)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            messages.append(response_message) # Adiciona resposta (ou pedido de ferramenta) ao histórico

            # 4. Loop de chamada de ferramentas (se necessário)
            while response_message.tool_calls:
                logger.info(f"[GreetingAgent] LLM solicitou {len(response_message.tool_calls)} ferramenta(s).")
                
                # Executa todas as ferramentas solicitadas
                tool_results = []
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name not in self.tool_implementations:
                        result = f"Erro: Ferramenta '{function_name}' desconhecida."
                    else:
                        try:
                            # Chama a função Python real (seja async ou sync)
                            function_to_call = self.tool_implementations[function_name]
                            
                            # Verifica se a função é async
                            if asyncio.iscoroutinefunction(function_to_call):
                                result = await function_to_call(**function_args)
                            else:
                                result = function_to_call(**function_args)
                                
                        except Exception as e:
                            logger.error(f"Erro ao executar ferramenta {function_name}: {e}")
                            result = f"Erro: {e}"
                    
                    # Adiciona o resultado da ferramenta para a próxima chamada do LLM
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(result), # Converte o resultado para string
                    })
                
                # Adiciona todos os resultados das ferramentas ao histórico
                messages.extend(tool_results)
                
                # 5. Segunda chamada à API (com os resultados das ferramentas)
                logger.info("[GreetingAgent] Chamando LLM novamente com resultados das ferramentas...")
                second_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                )
                response_message = second_response.choices[0].message
                messages.append(response_message) # Adiciona a nova resposta ao histórico

            # 6. Retorna a resposta final de texto
            logger.info("[GreetingAgent] LLM concluiu (sem mais ferramentas).")
            return response_message.content or "Nenhuma resposta gerada."

        except Exception as e:
            logger.error(f"Erro durante a invocação do GreetingAgent: {e}")
            return f"Desculpe, ocorreu um erro: {e}"