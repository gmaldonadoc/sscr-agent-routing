# =============================================================================
# agentes/greeting_agent/__main__.py
# =============================================================================
# 🎯 Propósito:
# Inicia o GreetingAgent como um servidor Agent-to-Agent (A2A).
# - Define os metadados do agente (AgentCard)
# - "Embrulha" (wraps) a lógica do GreetingAgent em um GreetingTaskManager
# - Escuta por tarefas recebidas em um host e porta configuráveis
# =============================================================================
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    
import logging                      # Módulo padrão do Python para logging
import click                        # Biblioteca para construir interfaces de linha de comando (CLI)

from utilities.a2a.agent_discovery import DiscoveryClient
from server.server import A2AServer   # Nossa implementação genérica de servidor A2A
from models.agent import (
    AgentCard,                      # Modelo Pydantic para descrever um agente
    AgentCapabilities,              # Descreve streaming e outros recursos
    AgentSkill                      # Descreve uma habilidade (skill) específica que o agente oferece
)
from .task_manager import GreetingTaskManager
                                    # TaskManager que adapta o GreetingAgent ao A2A
from .agent import GreetingAgent
                                    # Nossa lógica customizada do agente de orquestração

# -----------------------------------------------------------------------------
# ⚙️ Configuração de Logging
# -----------------------------------------------------------------------------
# Configura o logger raiz para imprimir mensagens de nível INFO no console.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# __name__ resolve para "agents.greeting_agent.__main__",
# então a saída deste logger será prefixada de acordo.

# -----------------------------------------------------------------------------
# ✨ Ponto de Entrada da CLI (Command-Line Interface)
# -----------------------------------------------------------------------------
@click.command()                    # Decorator: torna 'main' um comando de CLI
@click.option(
    "--host",                       # Nome da flag na CLI
    default="localhost",            # Valor padrão se a flag não for fornecida
    help="Host ao qual o servidor GreetingAgent será vinculado"  # Texto de ajuda para --help
)
@click.option(
    "--port",
    default=10001,                  # Porta padrão (deve bater com o agent_registry.json)
    help="Porta para o servidor GreetingAgent"
)
def main(host: str, port: int):
    """
    Inicia o servidor A2A do GreetingAgent.

    Args:
        host (str): Hostname ou IP ao qual se vincular (padrão: localhost)
        port (int): Porta TCP que o servidor ouvirá (padrão: 10001)
    """
    # Imprime um banner amigável para o usuário saber que o servidor está iniciando
    print(f"\n🚀 Iniciando GreetingAgent em http://{host}:{port}/\n")

    # -------------------------------------------------------------------------
    # 1) Define as capacidades (capabilities) do agente
    # -------------------------------------------------------------------------
    # Aqui especificamos que este agente NÃO suporta respostas via streaming.
    # Ele sempre enviará uma resposta única e completa.
    capabilities = AgentCapabilities(streaming=False)

    # -------------------------------------------------------------------------
    # 2) Define os metadados da habilidade (skill) do agente
    # -------------------------------------------------------------------------
    # Um AgentSkill descreve:
    # - id: identificador único para a habilidade
    # - name: nome legível
    # - description: o que esta habilidade faz
    # - tags: palavras-chave para descoberta ou categorização
    # - examples: exemplos de entradas do usuário para ilustrar a habilidade
    skill = AgentSkill(
        id="greet",                             # ID único da habilidade
        name="Greeting Tool",                   # Nome amigável
        description="Retorna uma saudação baseada na hora atual do dia",
        tags=["greeting", "time", "hello"],     # Tags pesquisáveis
        examples=["Greet me", "Say hello based on time"]  # Prompts de exemplo
    )

    # -------------------------------------------------------------------------
    # 3) Compõe o AgentCard para descoberta
    # -------------------------------------------------------------------------
    # O AgentCard é o metadado JSON que outros clientes/agentes
    # buscam de "/.well-known/agent.json". Ele descreve:
    # - nome, descrição, URL, versão
    # - modos de entrada/saída suportados
    # - capacidades e habilidades
    agent_card = AgentCard(
        name="GreetingAgent",                   # Identificador do Agente (usado no 'call_agent')
        description="Agente poeta que te cumprimenta baseado na hora atual",
        url=f"http://{host}:{port}/",            # URL base para descoberta
        version="1.0.0",                        # Versão semântica
        defaultInputModes=["text"],             # Aceita texto puro
        defaultOutputModes=["text"],            # Produz texto puro
        capabilities=capabilities,              # Streaming desabilitado
        skills=[skill]                          # Lista de habilidades
    )

    # -------------------------------------------------------------------------
    # 4) Instancia a lógica central e seu TaskManager
    # -------------------------------------------------------------------------
    # GreetingAgent contém a lógica de orquestração (LLM + ferramentas).
    greeting_agent = GreetingAgent()
    # GreetingTaskManager adapta essa lógica ao protocolo A2A JSON-RPC.
    task_manager = GreetingTaskManager(agent=greeting_agent)

    # -------------------------------------------------------------------------
    # 5) Cria e inicia o servidor A2A
    # -------------------------------------------------------------------------
    # O A2AServer conecta:
    # - Rotas HTTP (POST "/" para tarefas, GET "/.well-known/agent.json" para descoberta)
    # - Nossos metadados (AgentCard)
    # - O TaskManager que lida com as requisições recebidas
    server = A2AServer(
        host=host,
        port=port,
        agent_card=agent_card,
        task_manager=task_manager
    )
    server.start()  # Bloqueia aqui, servindo requisições até o processo ser interrompido


# -----------------------------------------------------------------------------
# Proteção de ponto de entrada
# -----------------------------------------------------------------------------
# Garante que `main()` só rode quando este script for executado diretamente,
# não quando for importado como um módulo.
if __name__ == "__main__":
    main()