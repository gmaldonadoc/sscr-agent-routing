# =============================================================================
# agentes/host_agent/entry.py
# =============================================================================
# 🎯 Propósito:
#   Inicializa o OrchestratorAgent como um servidor A2A.
#   Usa o DiscoveryClient para carregar os 'agent cards' dos agentes A2A filhos,
#   e então delega o roteamento (e as ferramentas MCP) para o OrchestratorAgent.
# =============================================================================
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import asyncio                      # Fornece ferramentas para trabalhar com código assíncrono
import logging                      # Módulo padrão do Python para registrar mensagens (info, warning, erros)
import click                        # Biblioteca de terceiros para construir interfaces de linha de comando (CLI)


from utilities.a2a.agent_discovery import DiscoveryClient  # Utilitário para descobrir outros agentes A2A via um registro JSON
from server.server import A2AServer   # A implementação central do servidor A2A (Starlette + JSON-RPC)
from models.agent import AgentCard, AgentCapabilities, AgentSkill  # Modelos Pydantic que descrevem metadados do agente
from agents.host_agent.orchestrator import (
    OrchestratorAgent,              # A lógica do orquestrador local (roteia tarefas)
    OrchestratorTaskManager         # Expõe o orquestrador sobre JSON-RPC
)

# Configura o logger raiz para exibir mensagens de nível INFO e acima
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)      # Cria uma instância de logger específica para este módulo


@click.command()                          # Declara esta função como um ponto de entrada de comando CLI
@click.option(
    "--host", default="localhost",
    help="Endereço ao qual o agente host será vinculado (bind)"   # Descrição para a flag --host
)
@click.option(
    "--port", default=10002, # <- Nota: Este provavelmente deveria ser 8000, e o TellTime 10002
    help="Porta para o agente host"        # Descrição para a flag --port
)
@click.option(
    "--registry", default=None,
    help=(
        "Caminho para o JSON de registro A2A. "
        "Padrão é 'utilities/a2a/agent_registry.json'"
    )
)
def main(host: str, port: int, registry: str):
    """
    Inicia o servidor A2A do OrchestratorAgent.

    Passos:
    1) Carrega os AgentCards dos filhos A2A via DiscoveryClient
    2) Instancia o OrchestratorAgent (com conectores A2A e ferramentas MCP)
    3) "Embrulha" (wrap) ele no OrchestratorTaskManager
    4) Inicia o servidor JSON-RPC
    """
    # 1) Descobre os agentes A2A filhos do arquivo de registro ou local padrão
    discovery = DiscoveryClient(registry_file=registry)
    # list_agent_cards() é async, então rodamos via asyncio.run
    # para obter o resultado de forma síncrona
    logger.info("Executando descoberta A2A (rede)...")
    agent_cards = asyncio.run(discovery.list_agent_cards())

    # Se nenhum agente for encontrado, avisa o usuário
    # (o orquestrador não terá alvos para delegar)
    if not agent_cards:
        logger.warning(
            "Nenhum agente A2A encontrado – o orquestrador não terá para quem delegar"
        )
    else:
        logger.info(f"Descoberta A2A concluída. {len(agent_cards)} agentes encontrados.")

    # 2) Define os metadados deste próprio agente host para descoberta por outros clientes
    capabilities = AgentCapabilities(streaming=False)  # Indica que este agente não suporta streaming
    skill = AgentSkill(
        id="orchestrate",                     # Identificador interno único para a habilidade
        name="Orchestrate Tasks",             # Nome amigável mostrado em UIs
        description=(
            "Roteia pedidos de usuário para agentes A2A filhos ou ferramentas MCP baseado na intenção."
        ),
        tags=["routing", "orchestration"],    # Palavras-chave para ajudar clientes a descobrir esta habilidade
        examples=[                            # Exemplos de consultas para ilustrar o uso
            "What is the time?",
            "Greet me",
            "Search the latest funding news for Acme Corp",
        ]
    )
    # Constrói o AgentCard, que é servido em /.well-known/agent.json
    orchestrator_card = AgentCard(
        name="OrchestratorAgent",              # Nome único do agente
        description="Delega para TellTimeAgent, GreetingAgent e ferramentas MCP",
        url=f"http://{host}:{port}/",           # Endpoint público onde este agente escuta
        version="1.0.0",                        # Versão semântica deste agente
        defaultInputModes=["text"],             # Modos de entrada suportados
        defaultOutputModes=["text"],            # Modos de saída suportados
        capabilities=capabilities,              # Capacidades de streaming
        skills=[skill]                          # Quais habilidades este agente fornece
    )

    # 3) Instancia a lógica do orquestrador e seu gerenciador de tarefas JSON-RPC
    logger.info("Inicializando o 'cérebro' do OrchestratorAgent (carregando ferramentas MCP)...")
    orchestrator = OrchestratorAgent(agent_cards=agent_cards)
    task_manager = OrchestratorTaskManager(agent=orchestrator)
    logger.info("Cérebro e TaskManager prontos.")

    # 4) Constrói e inicia o servidor A2A
    server = A2AServer(
        host=host,
        port=port,
        agent_card=orchestrator_card,
        task_manager=task_manager
    )
    
    logger.info(f"🚀 Servidor Orquestrador Host pronto e ouvindo em http://{host}:{port}/")
    server.start()  # Esta chamada bloqueia, rodando o servidor até ser interrompido


# Idioma padrão do Python: se este script for executado diretamente, chame main()
if __name__ == "__main__":
    main()