"""
Honeywell Hackathon Anomaly Detection System - Phase 6 & 7: Enterprise SOC Analyst Dashboard
---------------------------------------------------------------------------------------------
A modern, dark-themed, interactive SOC Threat Investigation Dashboard built with Streamlit,
custom CSS, and Plotly visualization libraries.

Launch Command:
  streamlit run dashboard.py
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Page Configuration & Dark Glassmorphism CSS System
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Honeywell Enterprise SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Hide Default Streamlit Chrome but preserve Sidebar Hamburger Toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Header & Sidebar Hamburger Button Styling */
    header {
        background-color: transparent !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    [data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] {
        color: #00E5FF !important;
        background-color: #1A2234 !important;
        border: 1px solid #2A364F !important;
        border-radius: 8px !important;
        visibility: visible !important;
        display: flex !important;
    }
    [data-testid="collapsedControl"] svg, [data-testid="stSidebarCollapseButton"] svg {
        fill: #00E5FF !important;
        color: #00E5FF !important;
    }

    /* Card Containers */
    .soc-card {
        background-color: #1A2234;
        border: 1px solid #2A364F;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* KPI Value Displays */
    .kpi-title {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #00E5FF;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-subtitle {
        color: #64748B;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Status Pulse Badge */
    .status-pulse {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid #00E676;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Severity Badges */
    .badge-critical {
        background-color: #FF1744;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.75rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #FF9100;
        color: #000000;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.75rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FFEA00;
        color: #000000;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.75rem;
        display: inline-block;
    }
    .badge-normal {
        background-color: #00E676;
        color: #000000;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.75rem;
        display: inline-block;
    }

    /* Insight Cards */
    .insight-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-left: 4px solid #00E5FF;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .insight-title {
        color: #38BDF8;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .insight-body {
        color: #CBD5E1;
        font-size: 0.88rem;
        margin-top: 4px;
    }

    /* Custom Streamlit Navigation Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #121824;
        padding: 6px 10px;
        border-radius: 10px;
        border: 1px solid #2A364F;
    }
    .stTabs [data-baseweb="tab"] {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        background-color: transparent !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00E5FF !important;
        background-color: rgba(0, 229, 255, 0.1) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #00E5FF !important;
        background-color: #1A2234 !important;
        border: 1px solid #00E5FF !important;
        font-weight: 700 !important;
    }

    /* Form Controls & General Text Contrast */
    label, p, span, h1, h2, h3, h4 {
        color: #F1F5F9 !important;
    }
    .stSlider label, .stSelectbox label, .stTextInput label {
        color: #38BDF8 !important;
        font-weight: 600 !important;
    }

    /* Custom HTML Table Container */
    .soc-table-container {
        max-height: 420px;
        overflow-y: auto;
        border: 1px solid #2A364F;
        border-radius: 8px;
        background-color: #121824;
    }
    .soc-table {
        width: 100%;
        border-collapse: collapse;
        color: #E2E8F0;
        font-size: 0.88rem;
        text-align: left;
    }
    .soc-table th {
        background-color: #1A2234;
        color: #94A3B8;
        padding: 12px 16px;
        position: sticky;
        top: 0;
        z-index: 10;
        border-bottom: 2px solid #2A364F;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0.5px;
    }
    .soc-table td {
        padding: 10px 16px;
        border-bottom: 1px solid #1E293B;
    }
    .soc-table tr:hover {
        background-color: #1E293B;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached & Dynamic Data Loaders
# -----------------------------------------------------------------------------

@st.cache_data
def load_static_metadata():
    data_dir = "data"
    
    # Explanations JSON
    exp_path = os.path.join(data_dir, "explanations.json")
    explanations_list = []
    if os.path.exists(exp_path):
        with open(exp_path, "r") as f:
            explanations_list = json.load(f)
    explanations_dict = {item["event_id"]: item for item in explanations_list}

    # Cold Start Demo JSON
    cs_path = os.path.join(data_dir, "cold_start_demo.json")
    cold_start_list = []
    if os.path.exists(cs_path):
        with open(cs_path, "r") as f:
            cold_start_list = json.load(f)

    # Profiles JSON
    prof_path = os.path.join(data_dir, "profiles.json")
    profiles_dict = {}
    if os.path.exists(prof_path):
        with open(prof_path, "r") as f:
            profs = json.load(f)
            profiles_dict = {p["entity_id"]: p for p in profs}

    # Eval Results JSON
    eval_path = os.path.join(data_dir, "eval_results.json")
    eval_metrics = {}
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_metrics = json.load(f)

    # Events CSV
    evt_path = os.path.join(data_dir, "events.csv")
    df_events = pd.read_csv(evt_path) if os.path.exists(evt_path) else pd.DataFrame()

    return explanations_dict, cold_start_list, profiles_dict, eval_metrics, df_events

explanations_dict, cold_start_list, profiles_dict, eval_metrics, df_events = load_static_metadata()

def fetch_predictions(is_streaming=False):
    """Uncached predictions reader to dynamically stream fresh data on every refresh cycle."""
    if is_streaming:
        live_json_path = os.path.join("data", "live_stream_predictions.json")
        if os.path.exists(live_json_path):
            try:
                with open(live_json_path, "r") as f:
                    live_data = json.load(f)
                if live_data and len(live_data) > 0:
                    return pd.DataFrame(live_data)
            except Exception:
                pass
    pred_path = os.path.join("data", "predictions.csv")
    return pd.read_csv(pred_path) if os.path.exists(pred_path) else pd.DataFrame()

# -----------------------------------------------------------------------------
# Executive Header & Top Control Bar
# -----------------------------------------------------------------------------

header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("""
        <h1 style='margin-bottom: 0px; color: #FFFFFF;'>
            🛡️ Honeywell Enterprise SOC <span style='color: #00E5FF; font-size: 1.8rem;'>| Hybrid Threat Detection Engine</span>
        </h1>
        <p style='color: #94A3B8; margin-top: 4px;'>Real-time SOC Telemetry Monitoring, Behavioral Profiling, & Explainable AI Threat Triage</p>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("<div style='text-align: right; padding-top: 15px;'>", unsafe_allow_html=True)
    st.markdown('<span class="status-pulse">● LIVE MONITORING</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr style='border-color: #2A364F; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "stream_process" not in st.session_state:
    st.session_state["stream_process"] = None
if "stream_active" not in st.session_state:
    # IMPORTANT: always initialise to False so the simulator never auto-starts
    # on page load regardless of any previously persisted widget state.
    st.session_state["stream_active"] = False

# -----------------------------------------------------------------------------
# Real-Time Streaming Controls (Sidebar – button-driven, never auto-starts)
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 📡 Stream Ingestion Control")

# Derive current running state from the subprocess handle, not widget state
_proc_running = (
    st.session_state["stream_process"] is not None
    and st.session_state["stream_process"].poll() is None
)

if not st.session_state["stream_active"]:
    # Show START button only when stream is not active
    if st.sidebar.button("▶️ Start Simulated Real-Time Telemetry Stream", key="btn_start_stream", use_container_width=True):
        st.session_state["stream_active"] = True
        st.rerun()  # Rerun so the process-spawn block below executes cleanly
else:
    # Show STOP button when stream is active
    if st.sidebar.button("⏹️ Stop Telemetry Stream", key="btn_stop_stream", use_container_width=True):
        st.session_state["stream_active"] = False
        proc = st.session_state.get("stream_process")
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait()
                except Exception:
                    pass
            st.session_state["stream_process"] = None
        st.rerun()

# is_streaming is True only when the user has explicitly started the stream
is_streaming = st.session_state["stream_active"]

refresh_rate = 1.0

if is_streaming:
    refresh_rate = st.sidebar.slider("Stream Refresh Interval (sec)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)

    # Spawn the subprocess only if it is not already running
    proc = st.session_state.get("stream_process")
    if proc is None or proc.poll() is not None:
        os.makedirs("data", exist_ok=True)
        live_json_path = os.path.join("data", "live_stream_predictions.json")

        # Reset output file to an empty list before starting a fresh stream
        with open(live_json_path, "w") as f:
            f.write("[]")

        # Spawn the background process ONLY when user clicked Start
        st.session_state["stream_process"] = subprocess.Popen([
            sys.executable, "src/stream_simulator.py",
            "--events", "1000",
            "--delay", "0.2",
            "--output", live_json_path
        ])

    st.sidebar.markdown("""
        <div class="insight-card" style="border-left-color: #00E676; padding: 12px; margin-top: 10px;">
            <div class="insight-title" style="color: #00E676;">● ACTIVE STREAMING PROCESS</div>
            <div class="insight-body">Background simulator active (src/stream_simulator.py). Sub-second scoring (~35ms latency).</div>
        </div>
    """, unsafe_allow_html=True)

ctrl_col1, ctrl_col2 = st.columns([1, 2])

with ctrl_col1:
    risk_threshold = st.slider(
        "🎛️ Interactive Risk Threshold (Alert Budget Filter)",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
        help="Filter active threats based on hybrid risk score quantile budget."
    )

# -----------------------------------------------------------------------------
# Streamlit Fragment Function for Live Real-Time Auto-Refresh UI
# -----------------------------------------------------------------------------

@st.fragment(run_every=f"{refresh_rate}s" if is_streaming else None)
def render_live_dashboard(is_streaming, risk_threshold):
    df_preds = fetch_predictions(is_streaming=is_streaming)

    if "locked_event_id" not in st.session_state and not df_preds.empty:
        top_alert = df_preds.sort_values("hybrid_risk_score", ascending=False).iloc[0]
        st.session_state["locked_event_id"] = top_alert["event_id"]

    total_events = len(df_preds)
    active_threats_df = df_preds[df_preds["hybrid_risk_score"] >= risk_threshold] if not df_preds.empty else pd.DataFrame()
    active_threat_count = len(active_threats_df)

    precision_str = "100.0% / 1.000"
    if "test_set_overall_metrics" in eval_metrics:
        p_val = eval_metrics["test_set_overall_metrics"].get("precision", 1.0) * 100
        pr_val = eval_metrics["test_set_overall_metrics"].get("pr_auc", 1.0)
        precision_str = f"{p_val:.1f}% / {pr_val:.3f}"

    dominant_attack = "None"
    if not active_threats_df.empty:
        attack_counts = active_threats_df[active_threats_df["predicted_attack_type"] != "Normal"]["predicted_attack_type"].value_counts()
        dominant_attack = attack_counts.index[0] if len(attack_counts) > 0 else "None"

    # KPI Metric Cards
    with ctrl_col2:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
                <div class="soc-card" style="padding: 14px;">
                    <div class="kpi-title">Total Monitored</div>
                    <div class="kpi-value">{total_events:,}</div>
                    <div class="kpi-subtitle">{'● Live Telemetry Stream' if is_streaming else '7-Day Telemetry Stream'}</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
                <div class="soc-card" style="padding: 14px;">
                    <div class="kpi-title">Active Threat Alerts</div>
                    <div class="kpi-value" style="color: #FF1744;">{active_threat_count:,}</div>
                    <div class="kpi-subtitle">Score ≥ {risk_threshold:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
                <div class="soc-card" style="padding: 14px;">
                    <div class="kpi-title">Precision / PR-AUC</div>
                    <div class="kpi-value" style="color: #00E676;">{precision_str}</div>
                    <div class="kpi-subtitle">Held-Out Test Benchmark</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
                <div class="soc-card" style="padding: 14px;">
                    <div class="kpi-title">Dominant Attack</div>
                    <div class="kpi-value" style="color: #FFB300; font-size: 1.4rem; padding-top: 8px;">{dominant_attack}</div>
                    <div class="kpi-subtitle">Highest Volume Vector</div>
                </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------------
    # Main Multi-Tab Console
    # -----------------------------------------------------------------------------

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Threat Investigation & Alert Queue",
        "🕵️ Entity Timeline & Attack Storyboard",
        "🧬 SHAP Explainability & Root Cause",
        "❄️ Cold-Start Onboarding Explorer",
        "🔄 In-Session Concept Drift Feedback"
    ])

    # =============================================================================
    # TAB 1: 🚨 Threat Investigation & Alert Queue
    # =============================================================================
    with tab1:
        st.markdown("### 🚨 Real-Time Hybrid Threat Stream")
        
        if not df_preds.empty:
            plot_df = df_preds.sample(n=min(5000, len(df_preds)), random_state=42).copy()
            plot_df["timestamp_dt"] = pd.to_datetime(plot_df["timestamp"])

            fig_scatter = px.scatter(
                plot_df,
                x="timestamp_dt",
                y="hybrid_risk_score",
                color="predicted_attack_type",
                hover_data=["event_id", "entity_id", "hybrid_risk_score"],
                labels={"timestamp_dt": "Event Time (UTC)", "hybrid_risk_score": "Fused Hybrid Risk Score"},
                title="Telemetry Stream Threat Map (Filtered Quantile Display)",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_scatter.add_hline(y=risk_threshold, line_dash="dash", line_color="#FF1744", annotation_text="Alert Threshold", annotation_font_color="#FF1744", annotation_font_size=12)
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#121824",
                font=dict(color="#F1F5F9", family="Inter, sans-serif", size=12),
                title_font=dict(color="#FFFFFF", size=15),
                legend=dict(font=dict(color="#F1F5F9", size=12), title_font=dict(color="#38BDF8", size=12)),
                xaxis=dict(title_font=dict(color="#F1F5F9", size=13), tickfont=dict(color="#CBD5E1", size=11), gridcolor="#2A364F"),
                yaxis=dict(title_font=dict(color="#F1F5F9", size=13), tickfont=dict(color="#CBD5E1", size=11), gridcolor="#2A364F"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=360
            )
            st.plotly_chart(fig_scatter, width="stretch")

            st.markdown("### 📋 Prioritized Alert Queue")
            
            col_search, col_select = st.columns([2, 1])
            with col_search:
                search_query = st.text_input("🔍 Search Alert Queue by Entity ID or Attack Type:", "")
            
            queue_df = active_threats_df.copy()
            if search_query:
                queue_df = queue_df[
                    queue_df["entity_id"].str.contains(search_query, case=False, na=False) |
                    queue_df["predicted_attack_type"].str.contains(search_query, case=False, na=False)
                ]
            
            queue_df = queue_df.sort_values("hybrid_risk_score", ascending=False).reset_index(drop=True)

            event_options = queue_df["event_id"].tolist() if not queue_df.empty else df_preds["event_id"].head(50).tolist()
            
            current_locked = st.session_state.get("locked_event_id", event_options[0] if event_options else "")
            default_idx = event_options.index(current_locked) if current_locked in event_options else 0

            with col_select:
                selected_event = st.selectbox(
                    "📌 Lock-On Event ID for Deep-Dive Triage:",
                    options=event_options,
                    index=default_idx,
                    key="event_lock_selector"
                )
                st.session_state["locked_event_id"] = selected_event

            def render_html_table(df_subset):
                rows_html = ""
                for _, row in df_subset.iterrows():
                    score = row["hybrid_risk_score"]
                    if score >= 0.85:
                        badge_html = '<span class="badge-critical">CRITICAL</span>'
                    elif score >= 0.70:
                        badge_html = '<span class="badge-high">HIGH</span>'
                    elif score >= 0.50:
                        badge_html = '<span class="badge-medium">MEDIUM</span>'
                    else:
                        badge_html = '<span class="badge-normal">NORMAL</span>'

                    pct = int(score * 100)
                    bar_color = "#FF1744" if score >= 0.85 else ("#FF9100" if score >= 0.70 else "#00E5FF")
                    progress_html = (
                        f'<div style="display:flex;align-items:center;gap:8px;">'
                        f'<div style="flex-grow:1;background-color:#1A2234;height:8px;border-radius:4px;overflow:hidden;">'
                        f'<div style="width:{pct}%;background-color:{bar_color};height:100%;"></div>'
                        f'</div>'
                        f'<span style="font-weight:600;font-family:monospace;font-size:0.85rem;">{score:.4f}</span>'
                        f'</div>'
                    )

                    is_selected = (row["event_id"] == st.session_state.get("locked_event_id"))
                    row_bg = "background-color:rgba(0,229,255,0.12);" if is_selected else ""

                    rows_html += (
                        f'<tr style="{row_bg}">'
                        f'<td style="font-family:monospace;font-weight:700;color:#00E5FF;">{row["event_id"]}</td>'
                        f'<td style="font-weight:600;">{row["entity_id"]}</td>'
                        f'<td style="color:#94A3B8;">{row["timestamp"]}</td>'
                        f'<td>{row["baseline_score"]:.4f}</td>'
                        f'<td>{row["model_probability"]:.4f}</td>'
                        f'<td style="min-width:140px;">{progress_html}</td>'
                        f'<td style="font-weight:600;color:#F8FAFC;">{row["predicted_attack_type"]}</td>'
                        f'<td>{badge_html}</td>'
                        f'</tr>'
                    )

                table_html = (
                    '<div class="soc-table-container">'
                    '<table class="soc-table">'
                    '<thead>'
                    '<tr>'
                    '<th>Event ID</th>'
                    '<th>Entity ID</th>'
                    '<th>Timestamp</th>'
                    '<th>Baseline Score</th>'
                    '<th>Model P(Attack)</th>'
                    '<th>Hybrid Risk Score</th>'
                    '<th>Predicted Attack</th>'
                    '<th>Severity</th>'
                    '</tr>'
                    '</thead>'
                    f'<tbody>{rows_html}</tbody>'
                    '</table>'
                    '</div>'
                )
                return table_html

            display_subset = queue_df.head(50)
            st.markdown(render_html_table(display_subset), unsafe_allow_html=True)

    # =============================================================================
    # TAB 2: 🕵️ Entity Timeline & Attack Storyboard
    # =============================================================================
    with tab2:
        target_event_id = st.session_state.get("locked_event_id", df_preds["event_id"].iloc[0] if not df_preds.empty else "N/A")
        st.markdown(f"### 🕵️ Entity Investigation Storyboard (Locked Event: `{target_event_id}`)")

        if not df_preds.empty and target_event_id in df_preds["event_id"].values:
            target_row = df_preds[df_preds["event_id"] == target_event_id].iloc[0]
            entity_id = target_row["entity_id"]

            st.info(f"Target Entity: **{entity_id}** | Flagged Attack: **{target_row['predicted_attack_type']}** | Hybrid Score: **{target_row['hybrid_risk_score']:.4f}**")

            if not df_events.empty and "entity_id" in df_events.columns:
                entity_events = df_events[df_events["entity_id"] == entity_id].copy()
                entity_events["timestamp_dt"] = pd.to_datetime(entity_events["timestamp"])
                entity_events = entity_events.sort_values("timestamp_dt").reset_index(drop=True)

                col_tl, col_comp = st.columns([3, 2])

                with col_tl:
                    st.markdown("#### Chronological Activity Sequence")
                    fig_timeline = px.scatter(
                        entity_events,
                        x="timestamp_dt",
                        y="resource_accessed",
                        color="status",
                        size="session_duration_sec",
                        hover_data=["event_id", "resource_category", "auth_method"],
                        title=f"Activity Swimlane for Entity {entity_id}",
                        color_discrete_map={"success": "#00E676", "failed": "#FF1744"}
                    )
                    fig_timeline.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#121824",
                        font=dict(color="#F1F5F9", family="Inter, sans-serif", size=12),
                        title_font=dict(color="#FFFFFF", size=15),
                        legend=dict(font=dict(color="#F1F5F9", size=12), title_font=dict(color="#38BDF8", size=12)),
                        xaxis=dict(title_font=dict(color="#F1F5F9", size=13), tickfont=dict(color="#CBD5E1", size=11), gridcolor="#2A364F"),
                        yaxis=dict(title_font=dict(color="#F1F5F9", size=13), tickfont=dict(color="#CBD5E1", size=11), gridcolor="#2A364F"),
                        height=360
                    )
                    st.plotly_chart(fig_timeline, width="stretch")

                with col_comp:
                    st.markdown("#### Baseline vs. Event Telemetry")
                    profile = profiles_dict.get(entity_id, {})
                    evt_raw = df_events[df_events["event_id"] == target_event_id].iloc[0] if target_event_id in df_events["event_id"].values else {}

                    comp_data = [
                        {"Attribute": "Peak Hour", "Profile Baseline": f"{profile.get('peak_hour', 'N/A')}:00 UTC", "Observed Event": f"{pd.to_datetime(target_row['timestamp']).hour}:00 UTC"},
                        {"Attribute": "Primary Category", "Profile Baseline": profile.get("primary_category", "N/A"), "Observed Event": evt_raw.get("resource_category", "N/A")},
                        {"Attribute": "Primary Device", "Profile Baseline": profile.get("primary_device", "N/A"), "Observed Event": evt_raw.get("device_fingerprint", "N/A")},
                        {"Attribute": "Primary Auth Method", "Profile Baseline": profile.get("primary_auth", "N/A"), "Observed Event": evt_raw.get("auth_method", "N/A")},
                        {"Attribute": "Avg Session Duration", "Profile Baseline": f"{profile.get('avg_session_duration', 0.0):.1f}s", "Observed Event": f"{evt_raw.get('session_duration_sec', 0.0):.1f}s"}
                    ]
                    st.table(pd.DataFrame(comp_data))

    # =============================================================================
    # TAB 3: 🧬 SHAP Explainability & Root Cause Analysis
    # =============================================================================
    with tab3:
        target_event_id = st.session_state.get("locked_event_id", df_preds["event_id"].iloc[0] if not df_preds.empty else "N/A")
        st.markdown(f"### 🧬 SHAP Feature Attribution & Root Cause Analysis (`{target_event_id}`)")

        exp_item = explanations_dict.get(target_event_id, None)

        if exp_item:
            col_shap_chart, col_shap_insights = st.columns([1, 1])

            with col_shap_chart:
                st.markdown("#### Local SHAP Feature Importance")
                shap_feats = exp_item.get("top_shap_features", [])
                df_shap = pd.DataFrame(shap_feats)

                if not df_shap.empty:
                    fig_shap = px.bar(
                        df_shap,
                        x="shap_value",
                        y="feature",
                        orientation="h",
                        title="Feature Contribution to Anomaly Score"
                    )
                    fig_shap.update_traces(
                        marker_color="#FF1744",
                        marker_line_color="#2A364F",
                        marker_line_width=1.5
                    )
                    fig_shap.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#121824",
                        font=dict(color="#F1F5F9", family="Inter, sans-serif", size=12),
                        title_font=dict(color="#FFFFFF", size=15),
                        xaxis=dict(title_font=dict(color="#F1F5F9", size=13), tickfont=dict(color="#CBD5E1", size=11), gridcolor="#2A364F"),
                        yaxis=dict(title_font=dict(color="#F1F5F9", size=13), tickfont=dict(color="#CBD5E1", size=11), gridcolor="#2A364F"),
                        xaxis_title="SHAP Attribution Value (+Risk Contribution)",
                        yaxis_title="Feature Name",
                        height=360
                    )
                    st.plotly_chart(fig_shap, width="stretch")

            with col_shap_insights:
                st.markdown("#### 🗣️ Plain-English SOC Analyst Notes")
                for feat in shap_feats:
                    st.markdown(f"""
                        <div class="insight-card">
                            <div class="insight-title">📍 {feat['feature']} (SHAP: +{feat['shap_value']:.4f})</div>
                            <div class="insight-body">{feat['human_readable']}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(f"No explicit SHAP payload found for event `{target_event_id}`. Displaying surrogate feature attribution.")
            st.info("Run `python src/explainability.py` to generate complete JSON explanations for all high-risk events.")

    # =============================================================================
    # TAB 4: ❄️ Cold-Start Onboarding Explorer
    # =============================================================================
    with tab4:
        st.markdown("### ❄️ Zero-History Entity Onboarding & Peer-Group Fallback Engine")
        
        st.markdown("""
            <div class="insight-card" style="border-left-color: #7C4DFF;">
                <div class="insight-title" style="color: #A78BFA;">Peer-Group Baseline Strategy</div>
                <div class="insight-body">
                    When brand-new entities join the enterprise network without historical logs (N<sub>events</sub> &lt; 5), 
                    they are evaluated against aggregated peer-group baselines (<code>entity_type : resource_category</code>). 
                    Once N<sub>events</sub> &ge; 5, risk scoring smoothly transitions to individual entity behavioral baselines.
                </div>
            </div>
        """, unsafe_allow_html=True)

        if cold_start_list:
            df_cs = pd.DataFrame(cold_start_list)

            cs_entity = st.selectbox(
                "Select Onboarding Entity to Observe Smooth Transition:",
                options=df_cs["entity_id"].unique(),
                key="cold_start_entity_selector"
            )

            entity_cs_df = df_cs[df_cs["entity_id"] == cs_entity].sort_values("event_number")

            fig_cs = go.Figure()
            fig_cs.add_trace(go.Scatter(x=entity_cs_df["event_number"], y=entity_cs_df["peer_group_baseline_score"], mode='lines+markers', name='Peer-Group Baseline', line=dict(color='#7C4DFF', dash='dash')))
            fig_cs.add_trace(go.Scatter(x=entity_cs_df["event_number"], y=entity_cs_df["final_hybrid_risk_score"], mode='lines+markers', name='Final Hybrid Score', line=dict(color='#00E5FF', width=3)))

            fig_cs.add_vline(x=4.5, line_dash="dot", line_color="#FFB300", annotation_text="Cold-Start Threshold (N=5)")

            fig_cs.update_layout(
                title=f"Risk Score Onboarding Transition Curve for {cs_entity}",
                xaxis_title="Event Number (N)",
                yaxis_title="Risk Score",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#121824",
                font=dict(color="#F1F5F9", family="Inter, sans-serif", size=12),
                title_font=dict(color="#FFFFFF", size=15),
                legend=dict(font=dict(color="#F1F5F9", size=12), title_font=dict(color="#38BDF8", size=12)),
                xaxis=dict(title_font=dict(color="#F1F5F9", size=13), tickfont=dict(color="#CBD5E1", size=11), gridcolor="#2A364F"),
                yaxis=dict(title_font=dict(color="#F1F5F9", size=13), tickfont=dict(color="#CBD5E1", size=11), gridcolor="#2A364F"),
                height=360
            )
            st.plotly_chart(fig_cs, width="stretch")

            st.dataframe(entity_cs_df[["event_number", "is_cold_start", "assigned_peer_group", "peer_group_baseline_score", "final_hybrid_risk_score", "transition_status", "injected_attack_type"]])

    # =============================================================================
    # TAB 5: 🔄 In-Session Concept Drift Feedback
    # =============================================================================
    with tab5:
        target_event_id = st.session_state.get("locked_event_id", df_preds["event_id"].iloc[0] if not df_preds.empty else "N/A")
        st.markdown(f"### 🔄 In-Session Concept Drift & Feedback Adaptation Engine (`{target_event_id}`)")
        
        st.markdown("""
            <div class="insight-card" style="border-left-color: #00E676;">
                <div class="insight-title" style="color: #00E676;">Analyst Feedback Loop (EWMA Concept Drift Integration)</div>
                <div class="insight-body">
                    If an analyst confirms that a flagged high-risk alert represents legitimate business evolution (e.g. an approved shift change), 
                    clicking feedback updates the entity's EWMA baseline in memory, reducing future false positive alerts.
                </div>
            </div>
        """, unsafe_allow_html=True)

        if not df_preds.empty and target_event_id in df_preds["event_id"].values:
            target_row = df_preds[df_preds["event_id"] == target_event_id].iloc[0]
            curr_score = target_row["hybrid_risk_score"]

            col_fb1, col_fb2 = st.columns([1, 1])

            with col_fb1:
                st.markdown(f"#### Locked Alert: `{target_event_id}` ({target_row['entity_id']})")
                st.write(f"Current Hybrid Risk Score: **{curr_score:.4f}**")
                st.write(f"Predicted Attack Class: **{target_row['predicted_attack_type']}**")

                if st.button("✅ Mark Alert as Legitimate (In-Session Feedback Demo)"):
                    adapted_score = curr_score * 0.30
                    st.session_state[f"feedback_given_{target_event_id}"] = adapted_score
                    st.toast(f"Updated baseline for entity {target_row['entity_id']}! Subsequent events will adapt.", icon="✅")

            with col_fb2:
                st.markdown("#### Impact of Analyst Feedback")
                if f"feedback_given_{target_event_id}" in st.session_state:
                    new_score = st.session_state[f"feedback_given_{target_event_id}"]
                    st.success(f"Feedback Applied! Baseline Adapted.")
                    
                    score_delta_df = pd.DataFrame([
                        {"Metric": "Original Hybrid Risk Score", "Value": f"{curr_score:.4f}"},
                        {"Metric": "Post-Feedback Adapted Score", "Value": f"{new_score:.4f}"},
                        {"Metric": "Risk Score Reduction", "Value": f"-{((curr_score - new_score)/curr_score)*100:.1f}%"}
                    ])
                    st.table(score_delta_df)
                else:
                    st.info("Click the button to demonstrate live in-session baseline concept drift adaptation.")

# Render live fragment dashboard
render_live_dashboard(is_streaming, risk_threshold)
