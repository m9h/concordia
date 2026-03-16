"""SustainHub FTC Hackathon Dashboard.

Shows the experiment ladder (RL vs Active Inference comparison)
and NVIDIA NIM model sweep results.

Usage:
    streamlit run hackathon_dashboard.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="SustainHub: Agentic Funding & Coordination",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Loading with Caching ---
@st.cache_data(ttl=60)
def load_json(path_or_url: str):
    if path_or_url.startswith("http"):
        import requests
        try:
            response = requests.get(path_or_url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error loading URL: {e}")
            return None
    elif os.path.exists(path_or_url):
        with open(path_or_url) as f:
            return json.load(f)
    return None

# --- Sidebar ---
with st.sidebar:
    st.header("Data Sources")
    ladder_path = st.text_input(
        "Experiment ladder results",
        "precomputed_ladder_results.json",
    )
    concordia_path = st.text_input(
        "Concordia results", "live_results.json"
    )
    if st.button("Force Refresh"):
        st.cache_data.clear()
        st.rerun()
    st.divider()

st.title("SustainHub: Open-Source Sustainability Through Active Inference")
st.markdown(
    "Comparing traditional RL agents with Active Inference agents in a "
    "social dilemma: maintain shared infrastructure or pursue individual rewards."
)

# =============================================================================
# TABS
# =============================================================================
tab_ladder, tab_sweep, tab_concordia, tab_research, tab_about = st.tabs([
    "Experiment Ladder", "Model Sweep", "Concordia Simulation",
    "BICA Research Brief", "About",
])

# =============================================================================
# TAB 1: Experiment Ladder
# =============================================================================
with tab_ladder:
    ladder_data = load_json(ladder_path)
    if not ladder_data:
        st.warning("No experiment data found. Run: `python -m examples.games.sustain_hub.research_loop --mode=quick`")
    else:
        st.subheader("RL to Active Inference: Progressive Ladder")

        # --- Summary metrics table ---
        has_std = "mean_hi_std" in ladder_data[0]
        summary_rows = []
        for r in ladder_data:
            row = {
                "Level": f"L{r['level']}",
                "Name": r["level_name"],
                "Mean HI": round(r.get("mean_hi", 0), 3),
                "Final HI": round(r.get("final_hi", 0), 3),
                "RQ": round(r.get("resilience_quotient", 0), 3),
                "Coverage": round(r.get("mean_coverage", 0), 3),
            }
            if has_std:
                row["HI \u00b1"] = round(r.get("mean_hi_std", 0), 3)
                row["RQ \u00b1"] = round(r.get("rq_std", 0), 3)
                row["Seeds"] = r.get("num_seeds", 1)
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # --- HI bar chart with error bars ---
        fig_bar = go.Figure()
        levels = [f"L{r['level']}" for r in ladder_data]
        mean_his = [r["mean_hi"] for r in ladder_data]
        std_his = [r.get("mean_hi_std", 0) for r in ladder_data]
        colors = ["#ef5350" if r["level"] < 4 else "#66bb6a" for r in ladder_data]

        fig_bar.add_trace(go.Bar(
            x=levels, y=mean_his,
            error_y=dict(type="data", array=std_his, visible=has_std),
            marker_color=colors,
            text=[f"{h:.3f}" for h in mean_his],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Mean Harmony Index by Level (red=RL, green=Active Inference)",
            yaxis_range=[0, 1], yaxis_title="Mean HI",
            template="plotly_dark", showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- HI trajectory with confidence bands ---
        st.subheader("Harmony Index Trajectories")
        all_levels = [f"L{r['level']}: {r['level_name']}" for r in ladder_data]
        default_sel = [all_levels[0], all_levels[3], all_levels[4], all_levels[-1]]
        selected_levels = st.multiselect(
            "Select levels to compare", all_levels,
            default=[s for s in default_sel if s in all_levels],
        )
        filtered_data = [
            r for r in ladder_data
            if f"L{r['level']}: {r['level_name']}" in selected_levels
        ]

        fig_traj = go.Figure()
        for r in filtered_data:
            hi_traj = r["hi_trajectory"]
            x = list(range(1, len(hi_traj) + 1))
            name = f'L{r["level"]}: {r["level_name"]}'
            fig_traj.add_trace(go.Scatter(
                x=x, y=hi_traj, name=name, mode="lines+markers",
            ))
            # Add confidence band if available
            if "hi_trajectory_std" in r:
                std = r["hi_trajectory_std"]
                upper = [h + s for h, s in zip(hi_traj, std)]
                lower = [h - s for h, s in zip(hi_traj, std)]
                fig_traj.add_trace(go.Scatter(
                    x=x + x[::-1],
                    y=upper + lower[::-1],
                    fill="toself", fillcolor="rgba(128,128,128,0.15)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))

        # Mark stress sprints
        fig_traj.add_vrect(x0=1.5, x1=2.5, fillcolor="red", opacity=0.08,
                           annotation_text="Stress", annotation_position="top left")
        fig_traj.add_vrect(x0=3.5, x1=4.5, fillcolor="red", opacity=0.08)
        fig_traj.update_layout(
            title="HI Over Sprints (shaded = stress events)",
            xaxis_title="Sprint", yaxis_title="Harmony Index",
            yaxis_range=[0, 1.05], template="plotly_dark",
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        # --- RQ comparison ---
        st.subheader("Resilience Quotient")
        rq_vals = [r.get("resilience_quotient", 1.0) for r in ladder_data]
        rq_stds = [r.get("rq_std", 0) for r in ladder_data]
        fig_rq = go.Figure()
        fig_rq.add_trace(go.Bar(
            x=levels, y=rq_vals,
            error_y=dict(type="data", array=rq_stds, visible=has_std),
            marker_color=colors,
            text=[f"{v:.3f}" for v in rq_vals],
            textposition="outside",
        ))
        fig_rq.add_hline(y=1.0, line_dash="dash", line_color="gray",
                         annotation_text="No degradation")
        fig_rq.update_layout(
            title="Resilience Quotient (>1 = improved under stress, <1 = degraded)",
            yaxis_title="RQ", template="plotly_dark",
        )
        st.plotly_chart(fig_rq, use_container_width=True)

# =============================================================================
# TAB 2: Model Sweep
# =============================================================================
with tab_sweep:
    st.subheader("LLM Backbone Comparison: NVIDIA NIM Model Sweep")
    sweep_dir = "/tmp/sustain_hub_sweep_v2"

    @st.cache_data(ttl=30)
    def load_sweep_results(sweep_path):
        results = []
        if not os.path.isdir(sweep_path):
            return results
        for d in sorted(os.listdir(sweep_path)):
            rf = os.path.join(sweep_path, d, "results.json")
            if os.path.isfile(rf):
                with open(rf) as f:
                    data = json.load(f)
                model_name = d.rsplit("_", 1)[0].replace("_", "/", 1)
                data["model"] = model_name
                results.append(data)
        return results

    sweep_results = load_sweep_results(sweep_dir)

    if not sweep_results:
        st.info("No sweep results yet. Run: `bash bin/nim_sweep.sh`")
    else:
        # Summary table
        df_sweep = pd.DataFrame([{
            "Model": r["model"],
            "HI": round(r["harmony_index"], 3),
            "RQ": round(r["resilience_quotient"], 3),
            "Agents": len(r.get("scores", {})),
            "Dropout": r.get("dropout_name", "N/A"),
        } for r in sweep_results]).sort_values("HI", ascending=False)
        st.dataframe(df_sweep, use_container_width=True, hide_index=True)

        # HI + RQ grouped bar chart
        fig_compare = go.Figure()
        models = df_sweep["Model"].tolist()
        fig_compare.add_trace(go.Bar(
            name="Harmony Index", x=models, y=df_sweep["HI"].tolist(),
            marker_color="#42a5f5",
        ))
        fig_compare.add_trace(go.Bar(
            name="Resilience Quotient", x=models, y=df_sweep["RQ"].tolist(),
            marker_color="#66bb6a",
        ))
        fig_compare.update_layout(
            barmode="group",
            title=f"HI and RQ by LLM Backbone ({len(df_sweep)} models)",
            yaxis_range=[0, 1.2], template="plotly_dark",
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # Sprint trajectories
        st.subheader("Sprint-by-Sprint Harmony Index")
        fig_sprint = go.Figure()
        for r in sweep_results:
            hi_vals = [s["harmony_index"] for s in r.get("sprint_history", [])]
            stress_markers = [s.get("stress") for s in r.get("sprint_history", [])]
            if hi_vals:
                fig_sprint.add_trace(go.Scatter(
                    x=list(range(1, len(hi_vals) + 1)),
                    y=hi_vals,
                    name=r["model"],
                    mode="lines+markers",
                ))
        fig_sprint.update_layout(
            xaxis_title="Sprint", yaxis_title="Harmony Index",
            yaxis_range=[0, 1], template="plotly_dark",
        )
        st.plotly_chart(fig_sprint, use_container_width=True)

# =============================================================================
# TAB 3: Concordia Simulation
# =============================================================================
with tab_concordia:
    results = load_json(concordia_path)
    if not results:
        st.info("No Concordia simulation results. Run a simulation first.")
    else:
        st.subheader("Concordia Simulation Results")

        # Key metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Harmony Index", f"{results['harmony_index']:.3f}")
        col2.metric("Resilience Quotient", f"{results['resilience_quotient']:.3f}")
        col3.metric("Policy", results.get("final_policy", "N/A"))

        # Sprint history table
        sprint_rows = []
        for i, s in enumerate(results.get("sprint_history", [])):
            row = {
                "Sprint": i + 1,
                "HI": round(s["harmony_index"], 3),
                "Stress": s.get("stress", "None"),
                "Policy": s.get("policy", ""),
            }
            # Add agent actions
            for name, action in s.get("joint_action", {}).items():
                score = s.get("scores", {}).get(name, 0)
                action_short = (action[:40] + "...") if action and len(action) > 40 else action
                row[name] = f"{action_short} ({score:+.0f})"
            sprint_rows.append(row)
        st.dataframe(pd.DataFrame(sprint_rows), use_container_width=True, hide_index=True)

        # HI trajectory
        hi_vals = [s["harmony_index"] for s in results.get("sprint_history", [])]
        if hi_vals:
            fig_hi = go.Figure()
            x = list(range(1, len(hi_vals) + 1))
            fig_hi.add_trace(go.Scatter(x=x, y=hi_vals, mode="lines+markers",
                                         name="Harmony Index"))
            # Color stress sprints
            for i, s in enumerate(results.get("sprint_history", [])):
                if s.get("stress"):
                    fig_hi.add_vrect(x0=i+0.5, x1=i+1.5, fillcolor="red",
                                     opacity=0.1)
            fig_hi.update_layout(
                title="Harmony Index Trajectory",
                xaxis_title="Sprint", yaxis_title="HI",
                yaxis_range=[0, 1], template="plotly_dark",
            )
            st.plotly_chart(fig_hi, use_container_width=True)

        # Agent scores
        scores = results.get("scores", {})
        roles = results.get("player_roles", {})
        if scores:
            score_df = pd.DataFrame([
                {"Agent": name, "Role": roles.get(name, "?"), "Score": score}
                for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ])
            fig_scores = px.bar(score_df, x="Score", y="Agent", color="Role",
                                orientation="h", title="Cumulative Agent Scores")
            fig_scores.update_layout(template="plotly_dark")
            st.plotly_chart(fig_scores, use_container_width=True)

# =============================================================================
# TAB 4: BICA Research Brief
# =============================================================================
with tab_research:
    st.subheader("Active Inference & Ostrom: Towards BICA 2026")
    if os.path.exists("docs/CLAUDE_RESEARCH_BRIEF.md"):
        with open("docs/CLAUDE_RESEARCH_BRIEF.md", "r") as f:
            st.markdown(f.read())
    else:
        st.info("Research brief not found in docs/")

# =============================================================================
# TAB 5: About
# =============================================================================
with tab_about:
    st.subheader("SustainHub: From Reward Maximization to Free Energy Minimization")
    st.markdown("""
**The Problem:** Open-source projects depend on volunteer contributions, but
face persistent sustainability challenges. Maintenance work (bug fixes, code
review, documentation) is underprovided because individual contributors are
incentivized to work on "flashy" features.

**The Approach:** We model OSS communities as multi-agent social dilemmas using
Google DeepMind's [Concordia](https://github.com/google-deepmind/concordia)
framework. LLM agents with distinct roles negotiate task allocation across
sprints while the project's health evolves.

**The Key Insight:** Traditional RL agents (Levels 0-3) optimize individual
reward and achieve ~52% mean Harmony Index with RQ=0.75 (degrade under stress).
Adding **Expected Free Energy** policy selection (Level 4+) pushes agents to
balance reward-seeking with information-seeking, achieving ~74% mean HI with
RQ>1.0 (improve under stress).

**What This Means for Funding the Commons:**
- Active Inference agents naturally balance exploration/exploitation
- EFE-based policies produce emergent cooperation without external incentives
- This suggests that funding mechanisms could be designed around *reducing
  uncertainty* (epistemic value) rather than just *maximizing output*
  (pragmatic value)

**References:**
- Rohira (2025). SustainHub: OSS Sustainability via Reinforcement Learning
- Parr, Pezzulo & Friston (2022). Active Inference: The Free Energy Principle
- Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active Inference
- Buterin, Hitzig & Weyl (2019). Quadratic Funding
- Ostrom (1990). Governing the Commons
""")
    st.markdown("---")
    st.markdown(
        "*Built for [Funding the Commons](https://www.fundingthecommons.io/ftc-frontiertower) "
        "Track 2: Agentic Funding & Coordination*"
    )
