# =============================================================================
# agentes/tell_time_agent/agent.py
# =============================================================================
# 🎯 Propósito:
# Este arquivo define um agente de IA muito simples chamado TellTimeAgent.
# (Adaptado para OpenAI) Ele usa o modelo da OpenAI para responder com a hora atual.
# =============================================================================


# -----------------------------------------------------------------------------
# 📦 Importações de Bibliotecas
# -----------------------------------------------------------------------------

from datetime import datetime     # Usado para obter a hora atual do sistema
from openai import OpenAI         # Cliente da OpenAI para o LLM
from dotenv import load_dotenv    # Para carregar variáveis de ambiente (como API keys)

# 🔐 Carrega variáveis de ambiente (como OPENAI_API_KEY) de um arquivo .env
load_dotenv() # Isso permite manter dados sensíveis fora do código.


# -----------------------------------------------------------------------------
# 🕒 TellTimeAgent: Seu agente de IA que informa as horas (com OpenAI)
# -----------------------------------------------------------------------------

class TellTimeAgent:
    """
    Este agente é "burro" de propósito. Seu único trabalho é responder
    com a hora atual, conforme instruído pelo seu prompt de sistema.
    """
    # Este agente suporta apenas entrada/saída de texto simples
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        """
        👷 Inicializa o TellTimeAgent:
        - Cria o cliente da OpenAI
        - Define o prompt do sistema (instrução)
        - Configura o armazenamento de sessão em memória
        """
        # 1. Inicializa o cliente OpenAI (lê a chave do .env)
        self.client = OpenAI()
        self.model = "gpt-4o" # Ou outro modelo da OpenAI
        
        # 2. Define o prompt de sistema (instrução)
        self.system_prompt = {
            "role": "system",
            "content": f"Você é um assistente de relógio. Responda apenas com a hora atual e nada mais. A hora atual é: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. NÃO converse, NÃO seja amigável, apenas informe a hora."
        }
        
        # 3. Armazenamento de sessão em memória (substitui o SessionService do ADK)
        # Mapeia session_id -> list[messages]
        self.sessions = {}

    async def invoke(self, query: str, session_id: str) -> str:
        """
        📥 Lida com uma consulta (query) do usuário e retorna uma string de resposta.
        
        Neste agente simples, a 'query' é ignorada, pois o prompt do sistema
        força o agente a sempre responder com a hora.

        Args:
            query (str): O que o usuário disse (ex: "que horas são?")
            session_id (str): Ajuda a agrupar mensagens em uma sessão

        Returns:
            str: A resposta do agente (a hora atual)
        """

        # 1. 🔁 Tenta reutilizar uma sessão existente (ou cria uma)
        if session_id not in self.sessions:
            # Cria um novo histórico de sessão com o prompt do sistema
            # NOTA: Recriamos o prompt aqui para que a hora seja ATUALIZADA
            # a cada nova sessão.
            system_prompt = {
                "role": "system",
                "content": f"Você é um assistente de relógio. Responda apenas com a hora atual. A hora atual é: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. NÃO converse, NÃO seja amigável, apenas informe a hora."
            }
            self.sessions[session_id] = [system_prompt]
        
        messages = self.sessions[session_id]
        
        # 2. 📨 Formata a mensagem do usuário
        messages.append({"role": "user", "content": query})

        # 3. 🚀 Executa o agente (chama a API da OpenAI)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            # Extrai a resposta do LLM
            agent_reply = response.choices[0].message
            
            # Adiciona a resposta do agente ao histórico
            messages.append(agent_reply)
            
            # 4. 📤 Retorna o conteúdo de texto da resposta
            return agent_reply.content or "Não foi possível obter a hora."

        except Exception as e:
            logger.error(f"Erro ao chamar a API da OpenAI no TellTimeAgent: {e}")
            return f"Erro ao processar a requisição: {e}"