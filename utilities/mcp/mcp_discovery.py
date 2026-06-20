# =============================================================================
# utilities/mcp/mcp_discovery.py
# =============================================================================
# 🎯 Propósito:
# Carrega um arquivo de configuração MCP (Model Context Protocol) listando um ou
# mais servidores MCP, e expõe uma API simples para recuperar suas definições.
# =============================================================================

import os                   # Fornece funções para interagir com o sistema de arquivos
import json                 # Fornece funções para analisar (parse) e gerar dados JSON
import logging              # Fornece capacidade de logging para avisos e erros
from typing import Dict, Any # Anotações de tipo: Dict[K, V] e tipo Any

# Cria um logger específico para este módulo
logger = logging.getLogger(__name__)


class MCPDiscovery:
    """
    🔍 Lê um arquivo de configuração JSON que define servidores MCP e fornece acesso
    às definições dos servidores sob a chave "mcpServers".

    Atributos:
        config_file (str): Caminho para o arquivo de configuração JSON.
        config (Dict[str, Any]): Conteúdo JSON analisado, esperado conter "mcpServers".
    """

    def __init__(self, config_file: str = None):
        """
        Inicializa o cliente de descoberta.

        Args:
            config_file (str, opcional): Caminho personalizado para o JSON de configuração do MCP.
                                         Se None, o padrão é 'mcp_config.json'
                                         localizado no mesmo diretório deste módulo.
        """
        # Se o chamador forneceu um config_file, use-o; senão, construa o caminho padrão
        if config_file:
            self.config_file = config_file  # Usa o caminho customizado
        else:
            # Determina o diretório deste módulo e junta com 'mcp_config.json'
            self.config_file = os.path.join(
                os.path.dirname(__file__),  # Diretório contendo este arquivo
                "mcp_config.json"           # Nome do arquivo de configuração padrão
            )

        # Carrega e analisa o arquivo de configuração JSON para um dict Python
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Lê e analisa o arquivo de configuração JSON.

        Returns:
            Dict[str, Any]: O objeto JSON inteiro se for válido;
                            caso contrário, um dict vazio em caso de erro.
        """
        try:
            # Tenta abrir o arquivo em modo de leitura 'r'
            with open(self.config_file, 'r') as f:
                data = json.load(f)  # Analisa (parse) o JSON para um objeto Python

            # Garante que o nível superior do JSON é um dicionário/objeto
            if not isinstance(data, dict):
                # Se não for, levanta um ValueError para acionar o manipulador de exceção
                raise ValueError("Configuração MCP deve ser um objeto JSON no nível superior.")

            # Retorna os dados JSON analisados
            return data

        except FileNotFoundError:
            # Loga um aviso se o arquivo não existir, depois retorna uma config vazia
            logger.warning(f"Arquivo de configuração MCP não encontrado: {self.config_file}")
            return {}

        except (json.JSONDecodeError, ValueError) as e:
            # Loga um erro se a análise (parse) falhar ou se os dados não forem um dict
            logger.error(f"Erro ao analisar (parse) a configuração MCP: {e}")
            return {}

    def list_servers(self) -> Dict[str, Any]:
        """
        Recupera o mapeamento de nomes de servidores para suas entradas de configuração.

        O JSON deve ter esta aparência:

        {
            "mcpServers": {
                "nome do servidor 1": { "command": "...", "args": [...] },
                "nome do servidor 2":   { "command": "...", "args": [...] }
            }
        }

        Returns:
            Dict[str, Any]: O dicionário sob "mcpServers", ou um dict vazio se ausente.
        """
        # Usa dict.get para recuperar com segurança 'mcpServers'; retorna {} se a chave não for encontrada
        return self.config.get('mcpServers', {})