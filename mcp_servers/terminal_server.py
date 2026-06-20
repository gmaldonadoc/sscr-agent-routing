# =============================================================================
# mcp_servers/terminal_server.py
# (VERSÃO CORRIGIDA - usando FastMCP)  ✅ retorna "OK" quando não há stdout
# =============================================================================

import os
import subprocess
import logging
import shlex
from typing import Optional

# Import correto para a sua versão da lib MCP
from mcp.server.fastmcp import FastMCP

# -----------------------------------------------------------------------------
# Configuração de logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MCPServer-Terminal")

# -----------------------------------------------------------------------------
# Instância FastMCP e Workspace
# -----------------------------------------------------------------------------
mcp = FastMCP("terminal")

# Define o diretório de trabalho (workspace) para os comandos de terminal
# 🔐 Dica: mantenha este diretório “sandboxado” para evitar sobrescrever arquivos do projeto por engano
#DEFAULT_WORKSPACE = os.path.expanduser("~/mcp/workspace")
DEFAULT_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(DEFAULT_WORKSPACE, exist_ok=True)
logger.info(f"Workspace do MCP Terminal definido para: {DEFAULT_WORKSPACE}")

# -----------------------------------------------------------------------------
# Utilitário: tentar inferir o nome de arquivo em comandos com redirecionamento
# -----------------------------------------------------------------------------
def _infer_redirect_filename(cmd: str) -> Optional[str]:
    """
    Tenta inferir o arquivo alvo quando há redirecionamento '>' ou '>>'.
    É um parser simples/heurístico (cobre casos comuns como: echo '...' > arquivo.txt).

    Retorna:
        - Nome do arquivo (string) se conseguir inferir; caso contrário, None.
    """
    # Normaliza espaços
    s = cmd.strip()

    # Procuramos o último '>>' primeiro para não confundir com '>'
    for sep in (">>", ">"):
        if sep in s:
            # Pega o pedaço após o redirecionador
            tail = s.split(sep, 1)[1].strip()
            # Se tiver pipe, fique só com a parte antes do próximo pipe
            tail = tail.split("|", 1)[0].strip()
            # Remove possíveis ; && || e pega só o primeiro token
            for breaker in ["&&", "||", ";"]:
                if breaker in tail:
                    tail = tail.split(breaker, 1)[0].strip()

            # remove aspas simples/duplas do começo/fim
            if (tail.startswith("'") and tail.endswith("'")) or (tail.startswith('"') and tail.endswith('"')):
                tail = tail[1:-1].strip()

            # por segurança, se houver espaço (ex: '>  caminho com espaço.txt'),
            # pegue apenas o primeiro “token” como nome do arquivo
            fname = tail.split()[0] if tail else ""
            return fname or None
    return None

# -----------------------------------------------------------------------------
# Ferramenta MCP: run_command
# -----------------------------------------------------------------------------
@mcp.tool(name="run_shell_command")
async def run_command(command: str) -> str:

    """
    Executa um comando de terminal dentro do diretório de trabalho (workspace).

    Args:
        command: Comando de shell a ser executado (ex: "echo 'Olá' > A2A_MCP_GREETING.txt").

    Returns:
        - stdout do comando, se houver; OU
        - mensagem "OK: ..." quando o comando terminar sem saída (caso típico de redirecionamento); OU
        - mensagem de erro amigável, se falhar.
    """
    logger.info(f"Recebido comando para executar: {command}")
    try:
        # ⚠️ Usamos shell=True para suportar redirecionamento (>, >>), pipes, etc.
        result = subprocess.run(
            command,
            shell=True,
            cwd=DEFAULT_WORKSPACE,
            capture_output=True,
            text=True,
            timeout=15  # um pouco maior para dar folga
        )

        if result.returncode != 0:
            logger.warning(f"Comando falhou com stderr: {result.stderr}")
            return f"Erro ao executar comando: {result.stderr.strip()}"

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        # ✅ Caso comum: redirecionamento gera stdout vazio → retornamos "OK" com caminho do arquivo (se possível)
        if not stdout and not stderr:
            inferred = _infer_redirect_filename(command)
            if inferred:
                file_path = os.path.join(DEFAULT_WORKSPACE, inferred)
                if os.path.exists(file_path):
                    msg = f"OK: comando concluído. Arquivo escrito em: {file_path}"
                else:
                    # Arquivo não encontrado (pode ter sido outro tipo de comando sem stdout)
                    msg = f"OK: comando concluído sem saída. Workspace: {DEFAULT_WORKSPACE}"
            else:
                msg = f"OK: comando concluído sem saída. Workspace: {DEFAULT_WORKSPACE}"

            logger.info(msg)
            return msg

        # Caso exista stdout (ex: comandos sem redirecionamento)
        out = stdout or f"[stderr]\n{stderr}"
        logger.info(f"Comando bem-sucedido. Saída (até 100 chars): {out[:100]}...")
        return out

    except subprocess.TimeoutExpired:
        logger.error("Timeout ao executar o comando.")
        return "Erro: tempo limite excedido ao executar o comando."
    except Exception as e:
        logger.error(f"Erro inesperado no run_command: {e}")
        return f"Erro inesperado: {e}"

# -----------------------------------------------------------------------------
# Bootstrap do servidor via stdio
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Iniciando MCPServer (Terminal) via Stdio com FastMCP...")
    mcp.run(transport="stdio")
