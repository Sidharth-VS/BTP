import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="TASR Admin Dashboard", page_icon="🛡️", layout="wide")

API_URL = os.getenv("COORDINATOR_API_URL", "http://coordinator:8000/api/v1/query").rstrip("/")
BASE_API_URL = (
    API_URL[: -len("/query")] if API_URL.endswith("/query") else API_URL.rsplit("/", 1)[0]
)
API_KEY = os.getenv("API_KEY", "fedrag-dev-key")
HEADERS = {"X-API-Key": API_KEY}
LOW_TRUST = 0.35

STATUS_ICON = {
    "healthy": "🟢",
    "degraded": "🟡",
    "unhealthy": "🔴",
    "rejected_sybil": "⛔",
}


@st.cache_data(ttl=3, show_spinner=False)
def fetch(endpoint: str, base_url: str, api_key: str) -> dict:
    response = requests.get(
        f"{base_url}/{endpoint}", headers={"X-API-Key": api_key}, timeout=8
    )
    response.raise_for_status()
    return response.json()


def fetch_optional(endpoint: str) -> dict:
    try:
        return fetch(endpoint, BASE_API_URL, API_KEY)
    except requests.exceptions.RequestException:
        return {}


def progress(value: float) -> None:
    value = max(0.0, min(float(value), 1.0))
    st.progress(value, text=f"{float(value):.4f}")


def histories_frame(histories: dict) -> pd.DataFrame:
    """Build a DataFrame from histories that may not share a length."""
    longest = max((len(values) for values in histories.values()), default=0)
    return pd.DataFrame(
        {
            name: list(values) + [None] * (longest - len(values))
            for name, values in histories.items()
        }
    )


st.title("🛡️ TASR Network Analytics")
st.caption(
    "Live trust components, evidence-feedback signals, cold-start schedule, and "
    "hardware-attestation state for every registered node."
)

top_left, top_mid, top_right = st.columns([1, 1, 4])
with top_left:
    refresh = st.button("🔄 Refresh", type="primary", width="stretch")
with top_mid:
    if st.button("🧹 Clear cache", width="stretch"):
        st.cache_data.clear()
        st.rerun()
with top_right:
    st.caption(
        f"Coordinator: `{BASE_API_URL}` · fetched {datetime.now().strftime('%H:%M:%S')}"
    )

if refresh:
    st.cache_data.clear()
    st.rerun()

try:
    nodes_payload = fetch("nodes", BASE_API_URL, API_KEY)
except requests.exceptions.RequestException as exc:
    st.error(
        "Could not reach the coordinator. Start it with "
        "`docker compose up -d` and check `/api/v1/health`."
    )
    st.code(str(exc))
    st.stop()

nodes = nodes_payload.get("nodes", [])
if not nodes:
    st.warning(
        "No nodes are registered yet. Start the node containers and wait for the "
        "first Flower health-check round."
    )
    st.stop()

summary = fetch_optional("tasr/summary")
attestation = fetch_optional("attestation/summary")
health = fetch_optional("health")
df = pd.DataFrame(nodes)

st.divider()

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
total_queries = int(summary.get("total_queries", 0) or 0)
warmup_queries = int(summary.get("warmup_queries", 5) or 5)
is_warmup = bool(summary.get("is_warmup_active", total_queries <= warmup_queries))

metrics = st.columns(5)
metrics[0].metric("Registered nodes", len(df))
metrics[1].metric(
    "Healthy", int((df.get("status") == "healthy").sum()) if "status" in df else 0
)
metrics[2].metric(
    f"Low trust (<{LOW_TRUST})",
    int((df["trust_score"] < LOW_TRUST).sum()) if "trust_score" in df else 0,
)
metrics[3].metric(
    "Average τ",
    f"{df['trust_score'].mean():.3f}" if "trust_score" in df else "—",
)
metrics[4].metric(
    "Minimum τ",
    f"{df['trust_score'].min():.3f}" if "trust_score" in df else "—",
)

progress_value = (
    min(total_queries / warmup_queries, 1.0) if warmup_queries > 0 else 1.0
)
st.progress(
    progress_value,
    text=(
        f"Warmup: {total_queries} / {warmup_queries} scored queries "
        f"({'complete' if not is_warmup else 'in progress'})"
    ),
)

overview_tab, nodes_tab, signals_tab, security_tab = st.tabs(
    ["📊 Overview", "🖥️ Nodes", "📈 Signals", "🔐 Security"]
)

with overview_tab:


    st.subheader("Node telemetry")
    table = df.copy()
    if "status" in table:
        table["status"] = table["status"].map(
            lambda value: f"{STATUS_ICON.get(str(value), '⚪')} {value}"
        )
    else:
        table["status"] = "unknown"
    if "trust_score" in table:
        table["warn"] = table["trust_score"].map(
            lambda value: "⚠️" if float(value) < LOW_TRUST else ""
        )
    else:
        table["warn"] = ""

    display_cols = [
        column
        for column in [
            "node_id",
            "trust_score",
            "u_rel",
            "u_cons",
            "u_agr",
            "s_i",
            "feedback_count",
        ]
        if column in table.columns
    ]
    display = table[display_cols].sort_values(
        "trust_score" if "trust_score" in display_cols else "node_id",
        ascending=False,
    )
    for column in ["trust_score", "u_rel", "u_cons", "u_agr", "s_i"]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{float(value):.4f}")
    st.dataframe(display, hide_index=True, width="stretch")

    st.download_button(
        "⬇️ Download node telemetry CSV",
        data=df.to_csv(index=False),
        file_name="fedrag_node_telemetry.csv",
        mime="text/csv",
    )

with nodes_tab:
    node_ids = sorted(df["node_id"].astype(str))
    selected_node = st.selectbox("Inspect a node", node_ids, key="node_inspect")
    node = df[df["node_id"].astype(str) == selected_node].iloc[0]

    st.subheader(f"`{selected_node}`")
    node_metrics = st.columns(6)
    node_metrics[0].metric("τ", f"{float(node.get('trust_score', 0.0)):.4f}")
    node_metrics[1].metric("u_rel", f"{float(node.get('u_rel', 1.0)):.4f}")
    node_metrics[2].metric("u_cons", f"{float(node.get('u_cons', 1.0)):.4f}")
    node_metrics[3].metric("u_agr", f"{float(node.get('u_agr', 1.0)):.4f}")
    node_metrics[4].metric("s_i", f"{float(node.get('s_i', 0.0)):.4f}")
    node_metrics[5].metric("feedback", int(node.get("feedback_count", 0)))

    bar_cols = st.columns(4)
    for column, key in zip(bar_cols, ["u_rel", "u_cons", "u_agr", "s_i"]):
        with column:
            st.caption(key)
            progress(float(node.get(key, 0.0)))

    st.caption(
        f"Domain: `{node.get('domain') or '—'}` · "
        f"status: `{node.get('status')}` · "
        f"attestation: `{node.get('attestation_status')}` · "
        f"fingerprint: `{str(node.get('hardware_fingerprint') or '—')[:24]}`"
    )
    if float(node.get("trust_score", 1.0)) < LOW_TRUST:
        st.warning(
            "This node is below the low-trust threshold and will be "
            "deprioritized by TASR routing."
        )

    with st.expander("Raw node record"):
        st.json(node.to_dict())

with signals_tab:
    node_ids = sorted(df["node_id"].astype(str))
    signal_node = st.selectbox("Node", node_ids, key="signal_node")
    node_summary = (summary.get("nodes") or {}).get(signal_node, {})

    trust_histories = {
        "u_rel": node_summary.get("reputation_history", []),
        "u_cons": node_summary.get("consistency_history", []),
        "u_agr": node_summary.get("agreement_history", []),
    }
    if any(trust_histories.values()):
        st.subheader("Trust trajectory")
        st.caption(
            "Per-query evolution of the three trust variables for the selected node."
        )
        st.line_chart(histories_frame(trust_histories), height=280)
    else:
        st.info(
            "No trust history yet. Histories are recorded from the first routed "
            "query (including warmup)."
        )

    feedback_histories = {
        "f_rel": node_summary.get("rel_feedback_history", []),
        "f_cons": node_summary.get("cons_feedback_history", []),
        "f_agr": node_summary.get("agr_feedback_history", []),
    }
    if any(feedback_histories.values()):
        st.subheader("Evidence feedback signals")
        st.caption(
            "Raw per-query feedback before thresholding. The update rule decays a "
            "signal when it is below the query's median or the node fails the "
            "relevance gate."
        )
        st.line_chart(histories_frame(feedback_histories), height=280)

    if node_summary:
        with st.expander("Raw TASR summary for this node"):
            st.json(node_summary)
    else:
        st.caption("The coordinator did not return a TASR summary for this node.")

with security_tab:
    if not attestation:
        st.info(
            "No attestation summary available. Is hardware attestation enabled?"
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Attestation", "ENABLED" if attestation.get("attestation_enabled") else "OFF"
        )
        col2.metric("Active hosts", attestation.get("active_hosts_count", 0))
        col3.metric(
            "Known fingerprints", attestation.get("total_known_fingerprints", 0)
        )
        col4.metric(
            "Sybil attempts blocked",
            attestation.get("total_sybil_attempts_blocked", 0),
        )

        active_hosts = attestation.get("active_hosts") or {}
        if active_hosts:
            st.subheader("Active hosts")
            st.json(active_hosts)
        else:
            st.caption("No active host leases.")

        st.caption(
            f"Max nodes per host: {attestation.get('max_nodes_per_host', '—')}"
        )

if health:
    st.caption(
        f"Coordinator health: `{health.get('status')}` · "
        f"{health.get('healthy_nodes', 0)}/{health.get('registered_nodes', 0)} "
        "healthy"
    )
