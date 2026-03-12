# SustainHub Session State - March 11, 2026

## Project Goal
Transforming SustainHub into a large-scale (16 agents), self-evolving community simulation using the Concordia framework.

## Key Changes Implemented

### 1. Infrastructure & Engine
*   **Modernization**: Created PEP 621 `pyproject.toml` for `uv` support.
*   **Termination Fix**: Concordia's default GMs often cap at 3 sprints. I implemented `CustomConversationGM` and `CustomDecisionGM` at the top level of `simulation.py` using a global `_CURRENT_PAYOFF` to override the `terminate` and `next_game_master` components.
*   **PayoffBasedTerminator**: A custom component that forces the simulation to run for exactly `num_sprints` by delegating `SceneTracker` methods.

### 2. Social & Cognitive Features
*   **Tools (`tools.py`)**: 
    *   `AutoCodeRover`: +15% success bonus on tasks when used.
    *   `ProjectStatsTool`: Allows agents to query the Harmony Index.
    *   `MentorshipTool`: Identifies junior members needing help.
*   **Community Size**: Expanded `AGENT_PROFILES` in `social_data.py` to 16 agents. Added `--community_size` flag to `run.py`.
*   **Evolutionary Governance**: Inserted a `RetrospectiveScene` every 3 sprints where agents vote on policies (Tool Subsidy, Maintenance Premium). Votes dynamically update the payoff rewards.

### 3. Metrics
*   **Harmony Index (HI)**: Balances productivity (success rate) vs. fairness (Gini coefficient of workload).
*   **Resilience Quotient (RQ)**: Measures HI recovery after contributor dropouts.

### 4. Runner Scripts
*   **`run.py`**: Cleaned up to support `num_sprints`, `community_size`, and `use_mock` flags.
*   **`sweep_runner.py`**: A parallel batch runner for Monte Carlo simulations.

## Current Status / Blockers
*   **Logic Verified**: Simulation works perfectly with `MockModel`.
*   **API Issue**: Encountering persistent `404 NOT_FOUND` errors when using `google-genai` SDK with model strings like `gemini-1.5-flash` or `gemini-2.0-flash-001`. 
*   **Next Step**: Resolve the model naming/SDK version conflict to run the 16-agent simulation on real LLMs.
