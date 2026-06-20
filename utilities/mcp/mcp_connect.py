# =============================================================================
# utilities/mcp/mcp_connect.py
# =============================================================================
# 🎯 Propósito:
#   Conectar-se a cada servidor MCP definido no mcp_config.json,
#   abrir sessões efêmeras para listar as ferramentas disponíveis, e
#   fornecer uma interface fácil para chamar essas ferramentas sob demanda.
# =============================================================================

import os                   # Para acessar variáveis de ambiente e caminhos de arquivo
import asyncio              # Para rodar funções assíncronas e loop de eventos
import logging              # Para registrar mensagens informativas e avisos
from dotenv import load_dotenv # Para carregar variáveis de ambiente de um arquivo .env

# Importa classes centrais do MCP para comunicação stdio e gerenciamento de sessão
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Utilitário local para ler a configuração dos servidores MCP
# (Este arquivo ainda não foi criado, mas é o próximo passo)
from utilities.mcp.mcp_discovery import MCPDiscovery

# Carrega variáveis de ambiente (ex: chaves de API) do .env para os.environ
load_dotenv()

# Cria um logger no nível do módulo usando o namespace do arquivo
logger = logging.getLogger(__name__)
# Configura o logger para exibir mensagens de nível INFO e acima
logging.basicConfig(level=logging.INFO)


class MCPTool:
    """
    🛠️ "Embrulha" (wrap) uma única ferramenta exposta pelo MCP para que
    possamos chamá-la facilmente.

    Atributos:
        name (str): Identificador da ferramenta (ex: "run_command").
        description (str): Descrição legível da ferramenta (para o LLM).
        input_schema (dict): Esquema JSON que define os argumentos esperados pela ferramenta.
        _params (StdioServerParameters): Comando/argumentos para iniciar o servidor MCP.
    """
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        server_cmd: str,
        server_args: list[str]
    ):
        # Armazena o nome e a descrição da ferramenta para referência futura
        self.name = name
        self.description = description
        # Salva o esquema JSON para validar os `args` passados para o run()
        self.input_schema = input_schema
        
        # Prepara os parâmetros de conexão stdio para que possamos
        # iniciar o servidor em cada chamada
        self._params = StdioServerParameters(
            command=server_cmd,
            args=server_args
        )

    async def run(self, args: dict) -> str:
        """
        Invoca a ferramenta ao:
         1. Iniciar o servidor MCP via stdio (em um novo processo)
         2. Inicializar uma ClientSession MCP
         3. Chamar a ferramenta (pelo nome) com os argumentos fornecidos
         4. Fechar a sessão automaticamente ao sair (graças ao 'async with')

        Retorna:
            O `content` (conteúdo) da resposta da ferramenta, ou a resposta
            bruta (como string) se 'content' não existir.
        """
        # Cria uma conexão stdio com o servidor MCP (sessão efêmera)
        # Isso EXECUTA o comando em self._params
        async with stdio_client(self._params) as (read_stream, write_stream):
            # "Embrulha" os streams stdio em uma ClientSession do MCP
            async with ClientSession(read_stream, write_stream) as sess:
                # Realiza qualquer "handshake" ou configuração inicial exigido pelo MCP
                await sess.initialize()
                # Chama a ferramenta no servidor com os argumentos fornecidos
                resp = await sess.call_tool(self.name, args)
                # Retorna o atributo `content` se presente, senão, a string da resposta
                return getattr(resp, "content", str(resp))


class MCPConnector:
    """
    🔗 Descobre servidores MCP a partir da configuração, lista as ferramentas de cada servidor,
    e as armazena em cache como instâncias de MCPTool para fácil consulta.

    Uso:
        connector = MCPConnector()
        tools = connector.get_tools()
        result = await tools[0].run({"arg1": "value"})
    """
    def __init__(self, config_file: str = None):
        # Inicializa o MCPDiscovery para carregar as definições do servidor do JSON
        self.discovery = MCPDiscovery(config_file=config_file)
        # Prepara uma lista vazia para guardar os objetos MCPTool
        self.tools: list[MCPTool] = []
        # Carrega as ferramentas de todos os servidores MCP configurados imediatamente
        self._load_all_tools()

    def _load_all_tools(self):
        """
        Método auxiliar interno: executa uma rotina assíncrona de forma síncrona
        para buscar e armazenar as definições de ferramentas de cada servidor MCP.
        """
        # Define a função assíncrona que faz o trabalho
        async def _fetch():
            # Obtém o mapeamento: nome do servidor -> seu dict de config
            servers = self.discovery.list_servers()
            # Itera através de cada entrada de servidor
            for name, info in servers.items():
                # Extrai o comando (ex: "python script.py") e os args
                cmd = info.get("command")
                args = info.get("args", [])
                logger.info(f"[MCPConnector] Buscando ferramentas do servidor MCP: {name}")
                # Prepara os parâmetros para o stdio_client
                params = StdioServerParameters(command=cmd, args=args)
                try:
                    # Abre uma conexão stdio com o servidor MCP (inicia o processo)
                    async with stdio_client(params) as (r, w):
                        # "Embrulha" em uma sessão de cliente para falar o protocolo MCP
                        async with ClientSession(r, w) as sess:
                            # Inicializa a sessão (handshake)
                            await sess.initialize()
                            # Pede ao servidor sua lista de ferramentas
                            tool_list = (await sess.list_tools()).tools
                            # Para cada ferramenta declarada, "embrulha" em um MCPTool
                            for t in tool_list:
                                self.tools.append(
                                    MCPTool(
                                        name=t.name,
                                        description=t.description,
                                        input_schema=t.inputSchema,
                                        # Importante: Armazena o comando para
                                        # que o .run() possa iniciar este servidor
                                        server_cmd=cmd, 
                                        server_args=args
                                    )
                                )
                            logger.info(
                                f"[MCPConnector] {len(tool_list)} ferramentas carregadas de {name}"
                            )
                except Exception as e:
                    # Se qualquer erro ocorrer (ex: servidor não disponível), loga um aviso
                    logger.warning(
                        f"[MCPConnector] Falha ao listar ferramentas de {name}: {e}"
                    )

        # Executa a corrotina assíncrona _fetch em um novo loop de eventos
        # (Isso torna o __init__ síncrono, como o Orchestrator espera)
        asyncio.run(_fetch())

    def get_tools(self) -> list[MCPTool]:
        """
        Retorna uma cópia superficial da lista de instâncias de MCPTool.
        Garante que código externo não possa modificar nosso cache interno.
        """
        return self.tools.copy()