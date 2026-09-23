import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TASR Admin Dashboard", page_icon="🛡️", layout="wide")

API_URL_NODES = os.getenv("COORDINATOR_API_URL", "http://coordinator:8000/api/v1/query").replace("/query", "/nodes")
API_KEY = os.getenv("API_KEY", "fedrag-dev-key")

st.title("🛡️ TASR Network Analytics")
st.markdown("Monitor real-time **Trust-Aware Secure Routing (TASR)** telemetry, node health, and trust decay across the federated network.")

# Manual refresh mechanism
if st.button("🔄 Refresh Telemetry"):
    st.rerun()

headers = {"X-API-Key": API_KEY}

try:
    response = requests.get(API_URL_NODES, headers=headers, timeout=10)
    response.raise_for_status()
    nodes_data = response.json().get("nodes", [])

    if not nodes_data:
        st.warning("No nodes currently registered in the network.")
    else:
        df = pd.DataFrame(nodes_data)
        
        # High-level Metrics
        healthy_count = len(df[df["status"] == "healthy"])
        degraded_count = len(df) - healthy_count
        avg_trust = df["trust_score"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Edge Nodes", len(df))
        col2.metric("Healthy Nodes", healthy_count)
        col3.metric("Degraded Nodes", degraded_count, delta_color="inverse")
        col4.metric("Avg Network Trust", f"{avg_trust:.3f}")

        st.markdown("---")

        # Trust Decay / Growth Visualization
        st.subheader("Live Node Trust Scores")
        st.caption("Nodes falling below a trust threshold (e.g., 0.5) are automatically isolated by the TASR router.")
        
        # Prepare data for plotting
        chart_data = df[["node_id", "trust_score"]].set_index("node_id")
        chart_data.sort_values(by="trust_score", ascending=False, inplace=True)
        
        # Render Bar Chart
        st.bar_chart(chart_data, color="#ff4b4b", height=400)

        # Detailed Routing Table
        st.subheader("Node Registry & Telemetry Details")
        display_df = df[["node_id", "status", "trust_score", "address", "last_seen"]].copy()
        display_df["trust_score"] = display_df["trust_score"].apply(lambda x: f"{x:.4f}")
        display_df.sort_values(by="trust_score", ascending=False, inplace=True)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

except requests.exceptions.RequestException as e:
    st.error(f"Failed to fetch telemetry from Coordinator: {e}")