# =============================================================================
# agentes/tell_time_agent/__main__.py
# =============================================================================
# 🎯 Propósito:
# Inicia o TellTimeAgent como um servidor Agent-to-Agent (A2A) autônomo.
# - Define os metadados do agente (AgentCard).
# - "Embrulha" (wraps) a lógica do TellTimeAgent em um TellTimeTaskManager.
# - Escuta por tarefas na porta 10002 (ou outra porta configurável).
# =============================================================================
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
from utilities.a2a.agent_discovery import DiscoveryClient
import sys
import os
import logging
import click
from dotenv import load_dotenv

# --- 1. Configuração de Path (ESSENCIAL) ---
# Adiciona o diretório raiz ao sys.path ANTES de qualquer importação de modelo.
# __file__ -> .../agents/tell_time_agent/__main__.py
# dirname(1) -> .../agents/tell_time_agent
# dirname(2) -> .../agents
# dirname(3) -> .../ (Raiz do Projeto)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- 2. Importações (Agora que o path está correto) ---
from server.server import A2AServer
from models.agent import (
    AgentCard, 
    AgentCapabilities, 
    AgentSkill
)
from models.task import TaskState # Usado no AgentCard

# Importa as classes específicas deste agente
from .agent import TellTimeAgent
from .task_manager import TellTimeTaskManager

# --- 3. Configuração de Ambiente e Logging ---

# Carrega o .env (necessário para a OPENAI_API_KEY)
# Assumindo que o .env está na pasta 'server/'
env_path = os.path.join(root_dir, 'server', '.env')
load_dotenv(dotenv_path=env_path)

# Configura o logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TellTimeAgent-Launcher")

# -----------------------------------------------------------------------------
# ✨ CLI Entrypoint
# -----------------------------------------------------------------------------
@click.command()
@click.option(
    "--host",
    default="localhost",
    help="Host ao qual o servidor TellTimeAgent será vinculado"
)
@click.option(
    "--port",
    default=10000, # Porta padrão (deve bater com o agent_registry.json)
    help="Porta para o servidor TellTimeAgent"
)
def main(host: str, port: int):
    """
    Inicia o servidor A2A do TellTimeAgent.
    """
    print(f"\n🚀 Iniciando TellTimeAgent em http://{host}:{port}/\n")

    # --- 1. Definir a Habilidade (Skill) do Agente ---
    # (O 'GreetingAgent' espera encontrar um agente com este nome/habilidade)
    skill = AgentSkill(
        id="get_current_time",
        name="TellTimeAgent", # O 'GreetingAgent' procura por 'TellTimeAgent'
        description="Retorna a data e hora atuais do servidor em uma string formatada.",
        tags=["time", "clock", "hora"],
        examples=["Que horas são?", "Qual é a data de hoje?"]
    )

    # --- 2. Definir as Capacidades (Capabilities) ---
    capabilities = AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=True # O InMemoryTaskManager suporta isso
    )

    # --- 3. Compor o AgentCard ---
    # Este é o JSON que será servido em /.well-known/agent.json
    agent_card = AgentCard(
        name="TellTimeAgent", # Nome principal que o GreetingAgent irá procurar
        description="Um agente 'robô' simples que informa a data e hora atuais.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=capabilities,
        skills=[skill] # Lista a habilidade que ele oferece
    )

    # --- 4. Instanciar o Cérebro e o TaskManager ---
    # 1. Cria o "cérebro" (a classe TellTimeAgent que usa OpenAI)
    agent_brain = TellTimeAgent()
    # 2. "Embrulha" (wraps) o cérebro no TaskManager
    task_manager = TellTimeTaskManager(agent=agent_brain)

    # --- 5. Criar e Iniciar o Servidor A2A ---
    server = A2AServer(
        host=host,
        port=port,
        agent_card=agent_card,
        task_manager=task_manager
    )
    
    logger.info(f"TellTimeAgent pronto e ouvindo em {agent_card.url}")
    logger.info(f"Endpoint de descoberta (para o Orquestrador): {agent_card.url}.well-known/agent.json")
    
    server.start() # Bloqueia aqui, servindo requisições


# --- Ponto de Entrada para Execução ---
if __name__ == "__main__":
    main()