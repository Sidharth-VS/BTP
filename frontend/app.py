import os
import requests
import streamlit as st

API_URL = os.getenv("COORDINATOR_API_URL", "http://coordinator:8000/api/v1/query")
API_KEY = os.getenv("API_KEY", "fedrag-dev-key")

st.set_page_config(page_title="FedRAG System Explorer", layout="wide")

st.title("🌐 FedRAG System Explorer")
st.markdown("Federated Retrieval-Augmented Generation across distributed ChromaDB nodes with local LLM synthesis.")

with st.sidebar:
    st.header("System Settings")
    st.text(f"Coordinator:\n{API_URL}")
    st.caption("Active Strategy: Broadcast + Central Ollama")

query = st.text_input("Enter your query:", placeholder="e.g., What are the standard medical protocols?")
col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.slider("Top-K per node", min_value=1, max_value=10, value=3)

if st.button("Search & Generate", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving from nodes and generating answer with Ollama..."):
            headers = {"X-API-Key": API_KEY}
            payload = {"query": query, "top_k": top_k}

            try:
                response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                data = response.json()

                # 1. Prominently display the synthesized answer
                st.subheader("🤖 Generated Answer")
                answer = data.get("answer")
                if answer:
                    st.success(answer)
                else:
                    st.info("No answer was generated.")

                # 2. Tabs for Context and Telemetry
                tab1, tab2 = st.tabs(["📚 Retrieved Federated Evidence", "🛰️ Routing Telemetry"])

                with tab1:
                    sources = data.get("sources", [])
                    if not sources:
                        st.info("No documents retrieved.")
                    for i, src in enumerate(sources):
                        node_id = src.get("node_id", "Unknown")
                        score = src.get("score", 0.0)
                        trust = src.get("trust_score", 1.0)

                        with st.expander(f"Source {i+1} | Node: `{node_id}` | Score: {score:.4f} | Trust: {trust}"):
                            st.write(src.get("content"))
                            st.caption(f"Metadata: {src.get('metadata')}")

                with tab2:
                    st.json(data.get("routing_info", {}))

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")