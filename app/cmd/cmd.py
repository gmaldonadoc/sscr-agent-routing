# =============================================================================
# app/cmd/cmd.py
# =============================================================================
# Propósito:
# Este arquivo é uma interface de linha de comando (CLI) que permite aos usuários
# interagir com o TellTimeAgent (ou qualquer agente) rodando em um servidor A2A.
#
# Ele envia mensagens de texto simples (como "What time is it?") para o agente,
# espera por uma resposta, e a exibe no terminal.
#
# Esta versão suporta:
# - envio básico de tarefas via A2AClient
# - reuso de sessão
# - impressão opcional do histórico da tarefa
# =============================================================================

import asyncclick as click      # click é uma ferramenta de CLI; asyncclick suporta funções async
import asyncio                  # Módulo nativo do Python para rodar loops de eventos async
from uuid import uuid4          # Usado para gerar IDs únicos de tarefa e sessão

# Importa o A2AClient do seu módulo client (ele lida com a lógica de requisição/resposta)
from client.client import A2AClient

# Importa o modelo Task para que possamos lidar e analisar (parse) as respostas do agente
from models.task import Task


# -----------------------------------------------------------------------------
# @click.command(): Transforma a função abaixo em um comando de linha de comando
# -----------------------------------------------------------------------------
@click.command()
@click.option("--agent", default="http://localhost:10002", help="URL base do servidor do agente A2A")
# ^ Define a opção --agent. É uma string com um padrão de localhost:10002
# ^ Usado para apontar para o servidor do agente em execução (ajuste se o servidor rodar em outro lugar)

@click.option("--session", default=0, help="ID da Sessão (use 0 para gerar uma nova)")
# ^ Define a opção --session. Uma sessão agrupa múltiplas tarefas.
# ^ Se o usuário passar 0, geramos um ID de sessão aleatório usando uuid4.

@click.option("--history", is_flag=True, help="Imprime o histórico completo da tarefa após receber a resposta")
# ^ Define uma flag --history (booleana). Se passada, o histórico completo da conversa é mostrado.

async def cli(agent: str, session: str, history: bool):
    """
    CLI para enviar mensagens de usuário para um agente A2A e exibir a resposta.

    Args:
        agent (str): A URL base do servidor do agente A2A (ex: http://localhost:10002)
        session (str): Ou um ID de sessão (string) ou 0 para gerar um
        history (bool): Se verdadeiro, imprime o histórico completo da tarefa
    """

    # Inicializa o cliente fornecendo o endpoint POST completo para enviar tarefas
    client = A2AClient(url=f"{agent}")

    # Gera um novo ID de sessão se não for fornecido (usuário passou 0)
    session_id = uuid4().hex if str(session) == "0" else str(session)

    # Inicia o loop de entrada principal
    while True:
        # Solicita a entrada do usuário
        prompt = await click.prompt("\nWhat do you want to send to the agent? (type ':q' or 'quit' to exit)")

        # Sai do loop se o usuário digitar ':q' ou 'quit'
        if prompt.strip().lower() in [":q", "quit"]:
            break

        # Constrói o payload (carga útil) usando o formato de tarefa JSON-RPC esperado
        payload = {
            "id": uuid4().hex,  # Gera um novo ID de tarefa único para esta mensagem
            "sessionId": session_id,  # Reusa ou cria o ID da sessão
            "message": {
                "role": "user",  # A mensagem é do usuário
                "parts": [{"type": "text", "text": prompt}]  # "Embrulha" (wrap) a entrada do usuário em uma text part
            }
        }

        try:
            # Envia a tarefa para o agente e obtém uma resposta Task estruturada
            task: Task = await client.send_task(payload)

            # Verifica se o agente respondeu (esperando pelo menos 2 mensagens: usuário + agente)
            if task.history and len(task.history) > 1:
                reply = task.history[-1]  # A última mensagem é geralmente do agente
                print("\nAgente diz:", reply.parts[0].text)  # Imprime a resposta em texto do agente
            else:
                print("\nNenhuma resposta recebida.")

            # Se a flag --history foi definida, mostra o histórico inteiro da conversa
            if history:
                print("\n========= Histórico da Conversa =========")
                for msg in task.history:
                    print(f"[{msg.role}] {msg.parts[0].text}")  # Mostra cada mensagem em sequência

        except Exception as e:
            import traceback
            traceback.print_exc()
            # Captura e imprime quaisquer erros (ex: servidor não está rodando, resposta inválida)
            print(f"\n❌ Erro ao enviar tarefa: {e}")


# -----------------------------------------------------------------------------
# Ponto de Entrada: Garante que a CLI rode apenas ao executar `python cmd.py`
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Roda a função async `cli()` dentro do loop de eventos
    asyncio.run(cli())