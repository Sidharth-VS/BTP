import json
import os
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("COORDINATOR_API_URL", "http://coordinator:8000/api/v1/query").rstrip("/")
BASE_API_URL = (
    API_URL[: -len("/query")] if API_URL.endswith("/query") else API_URL.rsplit("/", 1)[0]
)
API_KEY = os.getenv("API_KEY", "fedrag-dev-key")
HEADERS = {"X-API-Key": API_KEY}
TIMEOUT = 120

EXAMPLES = [
    ("Biology", "Why does sympathetic activity constrict pulmonary vessels?"),
    ("Security", "Why did TLS 1.3 drop AES-CBC?"),
    ("Travel", "Chennai or Pune for starting location on India Trip"),
    ("Medical", "What is cancer and why is early detection important?"),
]

st.set_page_config(page_title="FedRAG System Explorer", page_icon="🌐", layout="wide")

st.session_state.setdefault("query_text", "")
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("history", [])


@st.cache_data(ttl=5, show_spinner=False)
def fetch_health(base_url: str, api_key: str) -> dict:
    response = requests.get(
        f"{base_url}/health", headers={"X-API-Key": api_key}, timeout=4
    )
    response.raise_for_status()
    return response.json()


def render_answer(result: dict) -> None:
    data = result["data"]
    answer = data.get("answer")

    st.subheader("🤖 Synthesized Answer")
    if not result["use_llm"]:
        st.info("LLM synthesis is off — showing retrieved federated evidence only.")
    elif answer:
        with st.container(border=True):
            st.markdown(answer)
        st.caption(
            "Grounding context was ranked by **similarity × TASR trust (τ)** — a "
            "near-identical chunk from a low-trust node cannot outrank one from a "
            "trusted node."
        )
    else:
        st.info("No answer was generated.")


def render_evidence(sources: list, selected_nodes: list) -> None:
    if not sources:
        st.info("No evidence was returned for this query.")
        return

    selected = set(selected_nodes)
    by_node: dict[str, list[tuple[int, dict]]] = {}
    for rank, source in enumerate(sources, start=1):
        by_node.setdefault(str(source.get("node_id", "unknown")), []).append(
            (rank, source)
        )

    ordered = sorted(by_node.items(), key=lambda item: item[1][0][0])
    st.caption(
        f"{len(sources)} chunk(s) from {len(ordered)} node(s), ordered by "
        "similarity × trust. Chunks marked **primary** came from a node that "
        "was actually routed for generation."
    )

    for node_id, items in ordered:
        trust = max(float(source.get("trust_score", 0.0)) for _, source in items)
        tag = " · primary" if node_id in selected else ""
        with st.expander(
            f"`{node_id}` · τ {trust:.4f} · {len(items)} chunk(s){tag}",
            expanded=len(ordered) == 1,
        ):
            summary = pd.DataFrame(
                [
                    {
                        "Rank": rank,
                        "Similarity": round(float(source.get("score", 0.0)), 4),
                        "τ": round(float(source.get("trust_score", 0.0)), 4),
                        "Rank score": round(
                            float(
                                source.get(
                                    "rank_score",
                                    float(source.get("score", 0.0)) * trust,
                                )
                            ),
                            4,
                        ),
                        "Source": (source.get("metadata") or {}).get("source", "—"),
                    }
                    for rank, source in items
                ]
            )
            st.dataframe(summary, hide_index=True, width="stretch")

            for rank, source in items:
                st.markdown(f"**Rank {rank}**")
                st.write(source.get("content", ""))
                metadata = source.get("metadata") or {}
                if metadata:
                    st.caption(f"metadata: `{metadata}`")
                st.divider()


def render_routing(routing: dict) -> None:
    strategy = str(routing.get("strategy_used", "unknown")).upper()
    selected = routing.get("selected_nodes", [])
    trust_scores = {
        str(node): float(value)
        for node, value in (routing.get("trust_scores") or {}).items()
    }
    failed = routing.get("failed_nodes", [])

    st.markdown(f"**Strategy**: `{strategy}`")
    st.markdown(
        "**Answered nodes**: "
        + (", ".join(f"`{node}`" for node in selected) if selected else "—")
    )
    if failed:
        st.warning("Failed nodes: " + ", ".join(f"`{node}`" for node in failed))

    if trust_scores:
        records = [
            {"Node": node, "τ": round(value, 4)}
            for node, value in sorted(
                trust_scores.items(), key=lambda item: item[1], reverse=True
            )
        ]
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
        chart = pd.DataFrame(
            {"τ": [record["τ"] for record in records]},
            index=[record["Node"] for record in records],
        )
        st.bar_chart(chart, height=260)
    else:
        st.json(routing)


def render_result(result: dict) -> None:
    data = result["data"]
    routing = data.get("routing_info") or {}
    sources = data.get("sources") or []
    selected = routing.get("selected_nodes") or []

    st.divider()
    st.caption(
        f"Query: **{result['query']}** · {result['elapsed']:.2f}s · "
        f"top_k={result['top_k']} · {result['timestamp']}"
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Evidence chunks", len(sources))
    metric_cols[1].metric(
        "Nodes answered", routing.get("total_nodes_queried", len(selected))
    )
    metric_cols[2].metric("Strategy", str(routing.get("strategy_used", "unknown")).upper())
    metric_cols[3].metric("Latency", f"{result['elapsed']:.2f}s")

    render_answer(result)

    evidence_tab, routing_tab, raw_tab = st.tabs(
        ["📚 Retrieved Evidence", "🛰️ Routing & Trust", "🧾 Raw Response"]
    )
    with evidence_tab:
        render_evidence(sources, selected)
    with routing_tab:
        render_routing(routing)
    with raw_tab:
        st.download_button(
            "⬇️ Download response JSON",
            data=json.dumps(data, indent=2),
            file_name=f"fedrag_response_{result['stamp']}.json",
            mime="application/json",
            width="content",
        )
        st.json(data)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Coordinator")
    try:
        health = fetch_health(BASE_API_URL, API_KEY)
        st.success(
            f"Online · {health.get('healthy_nodes', 0)}/"
            f"{health.get('registered_nodes', 0)} nodes healthy"
        )
    except requests.exceptions.RequestException as exc:
        st.error("Unreachable")
        st.caption(f"`{BASE_API_URL}`\n\n{exc}")

    st.caption(f"API: `{BASE_API_URL}`")
    st.divider()

    st.header("Example queries")
    for index, (label, example) in enumerate(EXAMPLES):
        if st.button(label, key=f"example_{index}", width="stretch"):
            st.session_state.query_text = example
            st.rerun()

    st.divider()
    st.header("Session")
    if st.button("🧹 Clear results", width="stretch"):
        st.session_state.last_result = None
        st.session_state.history = []
        st.rerun()

    if st.session_state.history:
        st.caption("Recent queries")
        for index, item in enumerate(reversed(st.session_state.history[-5:])):
            if st.button(
                f"↺ {item['query'][:38]}",
                key=f"history_{index}",
                width="stretch",
            ):
                st.session_state.query_text = item["query"]
                st.rerun()


# ---------------------------------------------------------------------------
# Query form
# ---------------------------------------------------------------------------
st.title("🌐 FedRAG System Explorer")
st.markdown(
    "Federated Retrieval-Augmented Generation with **Trust-Aware Secure Routing "
    "(TASR)** and local LLM synthesis."
)

with st.form("query_form"):
    st.text_input(
        "Enter your query:",
        key="query_text",
        placeholder="e.g., What are the standard medical protocols?",
    )
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        top_k = st.slider("Top-K per node", min_value=1, max_value=10, value=3)
    with col2:
        use_llm = st.toggle(
            "Enable LLM synthesis",
            value=True,
            help="Turn off to return retrieved federated evidence without generating an answer.",
        )
    with col3:
        st.write("")
        submitted = st.form_submit_button(
            "🔍 Search & Generate", type="primary", width="stretch"
        )

if submitted:
    query = st.session_state.query_text.strip()
    if not query:
        st.warning("Please enter a query.")
    else:
        spinner_msg = (
            "Routing via TASR, querying nodes, and synthesizing answer..."
            if use_llm
            else "Routing via TASR and querying nodes (LLM synthesis disabled)..."
        )
        with st.spinner(spinner_msg):
            started = time.perf_counter()
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query, "top_k": top_k, "use_llm": use_llm},
                    headers=HEADERS,
                    timeout=TIMEOUT,
                )
                elapsed = time.perf_counter() - started
                if response.status_code >= 400:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.error(f"Coordinator returned {response.status_code}: {detail}")
                else:
                    data = response.json()
                    now = datetime.now()
                    timestamp = now.strftime("%H:%M:%S")
                    st.session_state.last_result = {
                        "data": data,
                        "query": query,
                        "stamp": now.strftime("%Y%m%d_%H%M%S"),
                        "elapsed": elapsed,
                        "use_llm": use_llm,
                        "top_k": top_k,
                        "timestamp": timestamp,
                    }
                    st.session_state.history.append(
                        {"query": query, "time": timestamp, "latency": elapsed}
                    )
            except requests.exceptions.Timeout:
                st.error(
                    f"The coordinator did not answer within {TIMEOUT}s. "
                    "Check that the nodes are connected and try again."
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"API request failed: {exc}")

if st.session_state.last_result:
    render_result(st.session_state.last_result)
