# =============================================================================
# 🎯 Propósito:
# Um módulo utilitário compartilhado para descobrir servidores Agente-para-Agente (A2A).
# Ele lê um registro de URLs base de agentes (de um arquivo JSON) e busca
# os metadados de cada agente (AgentCard) do endpoint de descoberta padrão.
# Isso permite que qualquer cliente ou agente aprenda dinamicamente sobre os agentes disponíveis.
# =============================================================================

import os             # os fornece funções para interagir com o sistema operacional, como caminhos de arquivo
import json           # json permite codificar e decodificar dados JSON
import logging        # logging é usado para registrar mensagens de aviso/erro/informação
from typing import List # List é uma dica de tipo (type hint) para funções que retornam listas

import httpx          # httpx é uma biblioteca cliente HTTP assíncrona para enviar requisições
from models.agent import AgentCard # AgentCard é um modelo Pydantic que representa os metadados de um agente

# Cria um logger nomeado para este módulo; __name__ é o nome do módulo
logger = logging.getLogger(__name__)


class DiscoveryClient:
    """
    🔍 Descobre agentes A2A lendo um arquivo de registro de URLs e consultando
    o endpoint /.well-known/agent.json de cada um para obter um AgentCard.

    Atributos:
        registry_file (str): Caminho para o arquivo JSON que lista as URLs base (strings).
        base_urls (List[str]): Lista carregada de URLs base dos agentes.
    """

    def __init__(self, registry_file: str = None):
        """
        Inicializa o DiscoveryClient.

        Args:
            registry_file (str, opcional): Caminho para o JSON de registro. Se None,
                o padrão é 'agent_registry.json' nesta pasta de utilitários.
        """
        # Se o chamador forneceu um caminho customizado, use-o; senão, construa o caminho padrão
        if registry_file:
            self.registry_file = registry_file
        else:
            # __file__ é o arquivo deste módulo; dirname obtém a pasta que o contém
            # join constrói um caminho para 'agent_registry.json' ao lado deste script
            self.registry_file = os.path.join(
                os.path.dirname(__file__),
                "agent_registry.json"
            )

        # Carrega imediatamente o arquivo de registro na memória
        self.base_urls = self._load_registry()

    def _load_registry(self) -> List[str]:
        """
        Carrega e analisa o arquivo JSON de registro para uma lista de URLs.

        Returns:
            List[str]: A lista de URLs base dos agentes, ou lista vazia em caso de erro.
        """
        try:
            # Abre o arquivo em self.registry_file no modo de leitura "r"
            with open(self.registry_file, "r") as f:
                # Analisa (parse) o arquivo inteiro como JSON
                data = json.load(f)
            # Garante que o JSON é uma lista, não um objeto ou outro tipo
            if not isinstance(data, list):
                raise ValueError("Arquivo de registro deve conter uma lista JSON de URLs.")
            return data
        except FileNotFoundError:
            # Se o arquivo não existe, loga um aviso e retorna uma lista vazia
            logger.warning(f"Arquivo de registro não encontrado: {self.registry_file}")
            return []
        except (json.JSONDecodeError, ValueError) as e:
            # Se o JSON é inválido ou do tipo errado, loga um erro e retorna uma lista vazia
            logger.error(f"Erro ao analisar (parse) arquivo de registro: {e}")
            return []

    async def list_agent_cards(self) -> List[AgentCard]:
        """
        Busca assincronamente o endpoint de descoberta de cada URL registrada
        e analisa o JSON retornado em objetos AgentCard.

        Returns:
            List[AgentCard]: Os agent cards recuperados com sucesso.
        """
        # Prepara uma lista vazia para coletar instâncias de AgentCard
        cards: List[AgentCard] = []

        # Cria um novo AsyncClient e garante que ele seja fechado ao terminar
        async with httpx.AsyncClient() as client:
            # Itera através de cada URL base no registro
            for base in self.base_urls:
                # Normaliza a URL (remove a barra final) e anexa o caminho de descoberta
                url = base.rstrip("/") + "/.well-known/agent.json"
                try:
                    # Envia uma requisição GET para o endpoint de descoberta com um timeout
                    response = await client.get(url, timeout=5.0)
                    # Lança uma exceção se o status da resposta for 4xx ou 5xx
                    response.raise_for_status()
                    # Converte a resposta JSON em um modelo Pydantic AgentCard
                    card = AgentCard.model_validate(response.json())
                    # Adiciona o AgentCard válido à nossa lista
                    cards.append(card)
                except Exception as e:
                    # Se algo der errado, loga qual URL falhou e o porquê
                    logger.warning(f"Falha ao descobrir agente em {url}: {e}")
        
        # Retorna a lista de AgentCards buscados com sucesso
        return cards