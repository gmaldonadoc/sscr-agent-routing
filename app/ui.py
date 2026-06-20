import os
import sys
import asyncio
import json
from uuid import uuid4

import streamlit as st

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from client.client import A2AClient


HOST_URL = "http://localhost:10002/"


async def send_query(query: str, session_id: str):
    client = A2AClient(url=HOST_URL)

    task = await client.send_task({
        "id": uuid4().hex,
        "sessionId": session_id,
        "message": {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "text": query
                }
            ]
        }
    })

    return task


def run_async(coro):
    return asyncio.run(coro)


st.set_page_config(
    page_title="Multi-Agent Structural Router",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 Multi-Agent Structural Router")
st.caption("Query → Normalizer → Tree → Graph → Constraint Filter → Pool → Orchestrator LLM")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuração")

    host_url = st.text_input("Host Agent URL", value=HOST_URL)

    if st.button("Nova sessão"):
        st.session_state.session_id = uuid4().hex
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Sessão")
    st.code(st.session_state.session_id)

    st.divider()
    st.subheader("Pipeline")
    st.markdown(
        """
        1. Request Normalizer  
        2. Hierarchical Routing Tree  
        3. Signal Graph Refinement  
        4. Constraint Filter  
        5. Compact Agent Pool  
        6. Orchestrator LLM  
        """
    )

col1, col2, col3 = st.columns([2, 1.2, 1.3])

with col1:
    st.subheader("Chat com o Host Agent")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Digite sua solicitação...")

    if query:
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Enviando para o OrchestratorAgent..."):
                try:
                    client = A2AClient(url=host_url)

                    async def call():
                        return await client.send_task({
                            "id": uuid4().hex,
                            "sessionId": st.session_state.session_id,
                            "message": {
                                "role": "user",
                                "parts": [
                                    {
                                        "type": "text",
                                        "text": query
                                    }
                                ]
                            }
                        })

                    task = run_async(call())
                    response_text = task.history[-1].parts[0].text

                    st.markdown(response_text)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })

                    st.session_state.last_task = task.model_dump()

                except Exception as e:
                    error_msg = f"Erro ao chamar Host Agent: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

with col2:
    st.subheader("Debug")

    st.markdown("**Host URL**")
    st.code(host_url)

    # ============================================================
    # Ferramentas usadas
    # ============================================================
    st.divider()
    st.subheader("Ferramentas usadas")

    if "last_task" in st.session_state:
        tools_used = (
            st.session_state.last_task
            .get("metadata", {})
            .get("tools_used", [])
        )

        if tools_used:
            for tool in tools_used:
                tool_name = tool.get("tool")

                if tool_name == "delegate_task":
                    st.success(f"Agente: {tool.get('agent')}")
                else:
                    st.info(f"Ferramenta: {tool_name}")
        else:
            st.warning("Nenhuma ferramenta registrada.")
    else:
        st.info("Nenhuma task enviada ainda.")

    # ============================================================
    # Structural Router Trace
    # ============================================================
    st.divider()
    st.subheader("Structural Router Trace")

    if "last_task" in st.session_state:
        routing_trace = (
            st.session_state.last_task
            .get("metadata", {})
            .get("routing_trace")
        )

        if routing_trace:
            with st.expander("1. Request Profile", expanded=True):
                st.json(routing_trace.get("request_profile", {}))

            with st.expander("2. Tree Candidates", expanded=True):
                st.json(routing_trace.get("tree_candidates", []))

            with st.expander("3. Graph Ranked", expanded=True):
                st.dataframe(
                    routing_trace.get("graph_ranked", []),
                    use_container_width=True
                )

            with st.expander("4. Constraint Filtered", expanded=True):
                st.dataframe(
                    routing_trace.get("constraint_filtered", []),
                    use_container_width=True
                )

            with st.expander("5. Compact Pool enviado para LLM", expanded=True):
                st.json(routing_trace.get("compact_pool", []))

            with st.expander("Resumo do roteamento", expanded=False):
                st.json(routing_trace.get("summary", {}))
        else:
            st.warning("Nenhum trace de roteamento encontrado.")
    else:
        st.info("Nenhuma task enviada ainda.")
        
with col3:
    st.subheader("Agent Cards + Prompt LLM")

    if "last_task" not in st.session_state:
        st.info("Nenhuma task enviada ainda.")
    else:
        metadata = st.session_state.last_task.get("metadata", {})

        agent_cards = metadata.get("agent_cards_snapshot", [])
        llm_pool_prompt = metadata.get("llm_pool_prompt")

        st.divider()
        st.markdown("### Agent Cards descobertos")

        if agent_cards:
            for card in agent_cards:
                with st.expander(card.get("name", "AgentCard"), expanded=False):
                    st.json(card)
        else:
            st.warning("Nenhum AgentCard encontrado no metadata.")

        st.divider()
        st.markdown("### Prompt enviado para a LLM")

        if llm_pool_prompt:
            st.text_area(
                "System prompt / pool compacto",
                value=llm_pool_prompt,
                height=500
            )
        else:
            st.warning("Nenhum prompt salvo.")
    # ============================================================
    # Última Task JSON
    # ============================================================
    st.divider()
    st.subheader("Última Task JSON")

    if "last_task" in st.session_state:
        st.json(st.session_state.last_task)
    else:
        st.info("Nenhuma task enviada ainda.")

    # ============================================================
    # Queries de teste
    # ============================================================
    st.divider()
    st.subheader("Queries de teste")

    examples = [
        "que horas são?",
        "oi, tudo bem?",
        "oi, você pode me dizer que horas são?",
        "gere uma saudação para mim usando os outros agentes e depois escreva a saudação em um arquivo chamado A2A_MCP_GREETING.txt usando a ferramenta run_shell_command",
    ]

    for example in examples:
        if st.button(example):
            st.session_state.messages.append({
                "role": "user",
                "content": example
            })
            st.rerun()