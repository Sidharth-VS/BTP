import os
import requests
import streamlit as st

# Read configuration from environment variables
API_URL = os.getenv("COORDINATOR_API_URL", "http://coordinator:8000/api/v1/query")
API_KEY = os.getenv("API_KEY", "fedrag-dev-key")

st.set_page_config(page_title="FedRAG System Explorer", layout="wide")

st.title("🌐 FedRAG System Explorer")
st.markdown("Query the federated nodes via **Broadcast Routing** + **Trust-Weighted Merge**.")

# Sidebar status
with st.sidebar:
    st.header("Connection Details")
    st.text(f"Endpoint:\n{API_URL}")
    st.caption("Authenticated via Bearer Header: `X-API-Key`")

# User Inputs
query = st.text_input("Enter your query:", placeholder="e.g., What are the standard medical protocols?")
col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.slider("Top-K per node", min_value=1, max_value=10, value=3)

# Search Execution
if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Broadcasting query to federated nodes..."):
            headers = {"X-API-Key": API_KEY}
            payload = {"query": query, "top_k": top_k}

            try:
                response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                data = response.json()

                st.success("Query completed successfully!")

                tab1, tab2 = st.tabs(["Retrieved Context", "TASR Routing Telemetry"])

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
                st.error(f"Coordinator API request failed: {e}")