"""SustainHub FTC Hackathon Dashboard.

Shows both the Concordia simulation results and the experiment ladder
(RL vs Active Inference comparison).

Usage:
    streamlit run hackathon_dashboard.py
"""

import json
import os
import requests

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
@st.cache_data(ttl=60)  # Refresh every minute
def load_json(path_or_url: str):
    if path_or_url.startswith("http"):
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
    st.info("Enter a local path or a raw URL (e.g., GitHub Gist)")
    concordia_path = st.text_input(
        "Concordia results", "live_results.json"
    )
    ladder_path = st.text_input(
        "Experiment ladder results",
        "precomputed_ladder_results.json",
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
# TAB 1: Experiment Ladder (RL vs AIF)
# =============================================================================
tab_ladder, tab_concordia, tab_about = st.tabs([
    "Experiment Ladder", "Concordia Simulation", "About"
])

with tab_ladder:
    ladder_data = load_json(ladder_path)
    if not ladder_data:
        st.warning(
            f"No experiment ladder data at `{ladder_path}`. "
            "Run: `python -m examples.games.sustain_hub.experiments --level=all`"
        )
    else:
        st.subheader("RL to Active Inference: Cumulative Concept Addition")

        # Summary metrics
        df = pd.DataFrame([
            {
                "Level": r["level"],
                "Name": r["level_name"],
                "Mean HI": r["mean_hi"],
                "Coverage": r["mean_coverage"],
                "Diversity": r["mean_diversity"],
                "Strategy Change": r["strategy_diversity"],
                "New Concept": r["new_concept"],
            }
            for r in ladder_data
        ])

        # Key insight metric cards
        rl_hi = df[df["Level"] <= 3]["Mean HI"].mean()
        aif_hi = df[df["Level"] >= 4]["Mean HI"].mean()
        rl_cov = df[df["Level"] <= 3]["Coverage"].mean()
        aif_cov = df[df["Level"] >= 4]["Coverage"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RL Mean HI (L0-3)", f"{rl_hi:.3f}")
        col2.metric("AIF Mean HI (L4-7)", f"{aif_hi:.3f}", f"{aif_hi - rl_hi:+.3f}")
        col3.metric("RL Task Coverage", f"{rl_cov:.0%}")
        col4.metric("AIF Task Coverage", f"{aif_cov:.0%}", f"{aif_cov - rl_cov:+.0%}")

        # Bar chart: HI by level
        fig_bar = px.bar(
            df,
            x="Name",
            y="Mean HI",
            color="Mean HI",
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            title="Mean Harmony Index by Experiment Level",
        )
        fig_bar.add_vline(
            x=3.5, line_dash="dash", line_color="white",
            annotation_text="EFE Policy Threshold",
            annotation_position="top left",
        )
        fig_bar.update_layout(
            xaxis_tickangle=-30,
            height=400,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # HI trajectories
        st.subheader("Harmony Index Trajectories Across Sprints")
        fig_traj = go.Figure()
        colors_rl = ["#ef4444", "#f97316", "#eab308", "#84cc16"]
        colors_aif = ["#22c55e", "#14b8a6", "#3b82f6", "#8b5cf6"]
        for r in ladder_data:
            hi_traj = r["hi_trajectory"]
            sprints = list(range(1, len(hi_traj) + 1))
            color = colors_rl[r["level"]] if r["level"] < 4 else colors_aif[r["level"] - 4]
            dash = "dot" if r["level"] < 4 else "solid"
            fig_traj.add_trace(go.Scatter(
                x=sprints,
                y=hi_traj,
                name=f'L{r["level"]}: {r["level_name"]}',
                line=dict(color=color, width=2, dash=dash),
                mode="lines+markers",
            ))
        fig_traj.update_layout(
            yaxis_title="Harmony Index",
            xaxis_title="Sprint",
            height=450,
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        # Conceptual bridge table
        with st.expander("Conceptual Bridge: RL to Active Inference"):
            for r in ladder_data:
                st.markdown(f"**Level {r['level']}: {r['level_name']}**")
                st.markdown(f"- RL analog: {r['rl_analog']}")
                st.markdown(f"- AIF mechanism: {r['aif_mechanism']}")
                st.markdown("---")

# =============================================================================
# TAB 2: Concordia Simulation
# =============================================================================
with tab_concordia:
    results = load_json(concordia_path)
    if not results:
        st.warning(
            f"No Concordia results at `{concordia_path}`. "
            "Run: `python -m examples.games.sustain_hub.run --project=... --fast`"
        )
    else:
        sprint_history = results.get("sprint_history", [])
        hi_history = [s["harmony_index"] for s in sprint_history]
        sprints = list(range(1, len(hi_history) + 1))

        # Summary cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Harmony Index", f"{results['harmony_index']:.3f}")
        col2.metric("Resilience Quotient", f"{results['resilience_quotient']:.3f}")
        col3.metric(
            "Agents",
            len(results.get("scores", {})),
        )

        # HI trajectory
        col_chart, col_scores = st.columns([2, 1])
        with col_chart:
            st.subheader("Community Health Trajectory")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sprints,
                y=hi_history,
                name="Harmony Index",
                line=dict(color="#22c55e", width=3),
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(34,197,94,0.1)",
            ))
            fig.add_hline(
                y=0.8, line_dash="dash", line_color="gray",
                annotation_text="Healthy threshold",
            )
            fig.update_layout(
                yaxis_range=[0, 1],
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_scores:
            st.subheader("Agent Scores")
            scores = results.get("scores", {})
            roles = results.get("player_roles", {})
            score_df = pd.DataFrame([
                {"Agent": name, "Role": roles.get(name, "?"), "Score": score}
                for name, score in sorted(
                    scores.items(), key=lambda x: x[1], reverse=True
                )
            ])
            st.dataframe(score_df, hide_index=True, use_container_width=True)

        # Sprint details
        st.subheader("Sprint-by-Sprint Details")
        for i, sprint in enumerate(sprint_history):
            with st.expander(
                f"Sprint {i+1} | HI = {sprint['harmony_index']:.3f}"
            ):
                # Task decisions
                actions = sprint.get("joint_action", {})
                sprint_scores = sprint.get("scores", {})
                rows = []
                for name, task in actions.items():
                    rows.append({
                        "Agent": name,
                        "Role": roles.get(name, "?"),
                        "Task": task[:60] if task else "None",
                        "Score": sprint_scores.get(name, 0.0),
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    use_container_width=True,
                )

                # Narrative if available
                narrative = results.get("narrative_history", [])
                if i < len(narrative) and narrative[i]:
                    st.markdown("**Active Inference Reasoning:**")
                    for agent_name, reason in narrative[i].items():
                        if isinstance(reason, dict):
                            st.markdown(
                                f"- **{agent_name}**: "
                                f"Uncertainty {reason.get('Uncertainty', '?')}/10 | "
                                f"Strategy: {reason.get('Strategy', '?')}"
                            )

# =============================================================================
# TAB 3: About
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
reward and achieve ~76% Harmony Index with 93% task coverage. Adding
**Expected Free Energy** policy selection (Level 4+) pushes agents to balance
reward-seeking with information-seeking, achieving ~83% HI with 100% coverage.

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
