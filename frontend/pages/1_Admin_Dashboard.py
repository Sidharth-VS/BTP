import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TASR Admin Dashboard", page_icon="🛡️", layout="wide")

BASE_API_URL = os.getenv("COORDINATOR_API_URL", "http://coordinator:8000/api/v1/query").replace("/query", "")
API_URL_NODES = f"{BASE_API_URL}/nodes"
API_URL_TASR_SUMMARY = f"{BASE_API_URL}/tasr/summary"
API_KEY = os.getenv("API_KEY", "fedrag-dev-key")

st.title("🛡️ TASR Network Analytics & Security Dashboard")
st.markdown(
    "Monitor real-time **Trust-Aware Secure Routing (TASR)** evidence feedback signals, "
    "reputation trajectories, cold-start schedule, and node isolation states."
)

if st.button("🔄 Refresh Telemetry", type="primary"):
    st.rerun()

headers = {"X-API-Key": API_KEY}

try:
    # Fetch node registry and TASR summary
    res_nodes = requests.get(API_URL_NODES, headers=headers, timeout=10)
    res_nodes.raise_for_status()
    nodes_data = res_nodes.json().get("nodes", [])

    tasr_summary = {}
    try:
        res_summary = requests.get(API_URL_TASR_SUMMARY, headers=headers, timeout=10)
        if res_summary.status_code == 200:
            tasr_summary = res_summary.json()
    except Exception:
        pass

    if not nodes_data:
        st.warning("No nodes currently registered in the network.")
    else:
        df = pd.DataFrame(nodes_data)

        # High-level Metrics Header
        healthy_count = len(df[df["status"] == "healthy"])
        avg_trust = df["trust_score"].mean() if "trust_score" in df else 1.0
        defense_mode = tasr_summary.get("defense_mode", "rel_cons_agr")
        total_queries = tasr_summary.get("total_queries", 0)
        warmup_queries = tasr_summary.get("warmup_queries", 50)
        is_warmup = tasr_summary.get("is_warmup_active", True)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Registered Nodes", len(df))
        col2.metric("Healthy Nodes", healthy_count)
        col3.metric("Avg Network Trust", f"{avg_trust:.3f}")
        col4.metric("Defense Mode", defense_mode.upper())
        warmup_label = "WARMUP" if is_warmup else "OPERATIONAL"
        col5.metric("Query Count", f"{total_queries} ({warmup_label})")

        st.markdown("---")

        # 1. Effective Trust Weight vs Component Scores Chart
        st.subheader("📊 Node Trust Score & Component Breakdown")
        st.caption(
            "Effective Weight τ_i = s_i · (u_rel)^α_r · g_c(u_cons) · g_a(u_agr). "
            "Nodes below the threshold are deprioritized or isolated."
        )

        cols_needed = ["node_id", "trust_score", "u_rel", "u_cons", "u_agr", "s_i"]
        avail_cols = [c for c in cols_needed if c in df.columns]
        chart_df = df[avail_cols].copy().set_index("node_id")

        st.bar_chart(chart_df, height=380)

        # 2. Detailed Node Telemetry Table
        st.subheader("📋 Node Telemetry Details")
        display_cols = ["node_id", "status", "domain", "trust_score", "u_rel", "u_cons", "u_agr", "s_i", "feedback_count"]
        display_cols = [c for c in display_cols if c in df.columns]
        display_df = df[display_cols].copy()

        # Format numerical metrics
        for col in ["trust_score", "u_rel", "u_cons", "u_agr", "s_i"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{float(x):.4f}")

        display_df.sort_values(by="trust_score" if "trust_score" in display_df else "node_id", ascending=False, inplace=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # 3. Trajectory History Visualizer (if history data exists)
        summary_nodes = tasr_summary.get("nodes", {})
        history_data = {}
        for nid, n_info in summary_nodes.items():
            hist = n_info.get("reputation_history", [])
            if hist:
                history_data[nid] = hist

        if history_data:
            st.subheader("📈 Historical Trust Trajectories (u_rel)")
            hist_df = pd.DataFrame(history_data)
            st.line_chart(hist_df, height=300)

except requests.exceptions.RequestException as e:
    st.error(f"Failed to fetch telemetry from Coordinator: {e}")