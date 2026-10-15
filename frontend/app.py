import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("COORDINATOR_API_URL", "http://coordinator:8000/api/v1/query")
API_KEY = os.getenv("API_KEY", "fedrag-dev-key")

st.set_page_config(page_title="FedRAG System Explorer", page_icon="🌐", layout="wide")

st.title("🌐 FedRAG System Explorer")
st.markdown(
    "Federated Retrieval-Augmented Generation with **Trust-Aware Secure Routing (TASR)** "
    "and local LLM synthesis."
)

query = st.text_input("Enter your query:", placeholder="e.g., What are the standard medical protocols?")
col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.slider("Top-K per node", min_value=1, max_value=10, value=3)
with col2:
    use_llm = st.toggle(
        "Enable LLM synthesis",
        value=True,
        help="Turn off to return retrieved federated evidence without generating an answer.",
    )

if st.button("🔍 Search & Generate", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        spinner_msg = (
            "Routing via TASR, querying nodes, and synthesizing answer..."
            if use_llm
            else "Routing via TASR and querying nodes (LLM synthesis disabled)..."
        )
        with st.spinner(spinner_msg):
            headers = {"X-API-Key": API_KEY}
            payload = {"query": query, "top_k": top_k, "use_llm": use_llm}

            try:
                response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                data = response.json()

                # 1. Prominently display synthesized answer
                st.subheader("🤖 Synthesized Answer")
                answer = data.get("answer")
                if not use_llm:
                    st.info("LLM synthesis is off — showing retrieved federated evidence only.")
                elif answer:
                    st.success(answer)
                    st.caption(
                        "Grounding context was ranked by **similarity × TASR trust (τ_i)**, "
                        "not similarity alone — so a near-identical chunk from a low-trust "
                        "node cannot outrank one from a trusted node."
                    )
                else:
                    st.info("No answer was generated.")

                # 2. Tabs for Context & Telemetry
                tab1, tab2 = st.tabs(["📚 Retrieved Federated Evidence", "🛰️ TASR Routing Telemetry"])

                with tab1:
                    sources = data.get("sources", [])
                    if not sources:
                        st.info("No documents retrieved.")
                    # The API returns these already ordered by similarity × trust.
                    for i, src in enumerate(sources):
                        node_id = src.get("node_id", "Unknown")
                        score = src.get("score", 0.0)
                        trust = src.get("trust_score", 1.0)
                        rank = src.get("rank_score", score * trust)

                        with st.expander(
                            f"Rank {i+1} | Node: `{node_id}` | "
                            f"Rank Score (sim×τ): {rank:.4f} | "
                            f"Similarity: {score:.4f} | TASR Trust (τ): {trust:.4f}"
                        ):
                            st.write(src.get("content"))
                            st.caption(f"Metadata: {src.get('metadata')}")

                    st.caption(
                        "Rank Score = similarity × TASR trust (τ_i). Ordering here is the "
                        "same order the LLM sees, so a high-similarity chunk from a "
                        "decayed node is demoted."
                    )

                with tab2:
                    r_info = data.get("routing_info", {})
                    st.markdown(f"**Strategy Used**: `{r_info.get('strategy_used', 'tasr').upper()}`")
                    st.markdown(f"**Selected Target Nodes**: `{r_info.get('selected_nodes', [])}`")
                    st.markdown(f"**Total Nodes Queried**: `{r_info.get('total_nodes_queried', 0)}`")

                    st.subheader("Current Node Trust Weights (τ_i)")
                    t_scores = r_info.get("trust_scores", {})
                    if t_scores:
                        t_df = pd.DataFrame(
                            [{"Node ID": k, "TASR Trust Weight (τ_i)": float(v)} for k, v in t_scores.items()]
                        )
                        st.dataframe(t_df, use_container_width=True, hide_index=True)
                    else:
                        st.json(r_info)

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")