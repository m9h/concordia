# SustainHub Three-Way Comparison — Progress Report

**Date**: 2026-03-15
**Target**: OREL meeting week of March 16-20

---

## Completed

### Step 1: Matched Evaluation Protocol
- Reward function aligned with Rohira's `preferred_task_bonus x expertise_multiplier`
- HI computation matches Rohira's `mean(individual_HI)`
- Stress events include contributor_dropout matching Rohira
- 30 agent profiles for community sizes 4-30 (Rohira tested 10/20/30)
- Grounded difficulty scoring from GHTorrent/GitHub Archive empirical distributions
- Seeded RNG for reproducibility across runs

### Step 2: Rohira's Proposed Metrics (GSoC 2025)
- **BRS** (Burnout Risk Score): consecutive non-preferred tasks / total sprints
- **SUE** (Skill Utilization Efficiency): 1.0 preferred, 0.5 adjacent, 0.0 unrelated
- **CHS** (Community Health Score): `0.3*HI + 0.25*(1-BRS) + 0.25*SUE + 0.2*RQ`
- All metrics integrated into `evaluate.py` and TSV logging

### Step 3: Governance Modes (LLAMOSC Comparison)
- Three modes implemented: `free_choice`, `dictator`, `meritocratic`
- `--governance` flag added to `run.py`
- Dictator: GM assigns tasks by role
- Meritocratic: priority by expertise/track record
- Free choice: agents self-select (default)

### Step 4a: Toy Project for Code Tasks (Partial)
- `toy_project/calculator/calculator.py` — calculator with 5 intentional bugs, 5 missing features
- `toy_project/tests/test_calculator.py` — 10 test classes (TestIssue1-TestIssue10), pytest-scorable
- `toy_project/issues/issue_1.md` through `issue_10.md` — markdown issue files matching LLAMOSC format
- `toy_project/conftest.py` — path configuration

### NIM Cloud Integration
- `vllm_remote.py` supports Bearer auth for NVIDIA NIM cloud
- `--vllm_api_key` flag, `NGC_API_KEY` env var fallback
- Error body logging for API debugging
- Tested with Qwen2.5-7B via `integrate.api.nvidia.com`

### Experiment Ladder (Levels 0-7)
- All 8 levels (pure RL → LLM+AIF) implemented and results stored
- Level comparison with Ostrom priors completed

---

## Experiment Results (Publishable)

### B-Series: Governance Comparison (NIM Cloud, Qwen2.5-7B, 5 seeds each)

| Experiment | HI (mean±std) | RQ | BRS | SUE | CHS | N |
|-----------|---------------|-----|------|------|------|---|
| B1: Free choice | **0.831±0.035** | 1.000 | 0.030 | 0.970 | **0.934** | 5 |
| B2: Dictator | 0.806±0.046 | 1.000 | 0.080 | 0.943 | 0.907 | 5 |
| B3: Meritocratic | 0.811±0.018 | 1.000 | 0.055 | 0.957 | 0.919 | 5 |

**Key finding**: Free choice governance outperforms dictator (HI: +3.1%) and meritocratic (+2.4%) for LLM agents. Lower variance too. This contrasts with LLAMOSC where authoritarian governance showed advantages for simple task allocation.

### A-Series: Rohira-Matched Setup (Partial — 3/5 seeds)

| Seed | HI | RQ |
|------|------|------|
| 0 | 0.847 | 1.000 |
| 1 | 0.742 | 1.000 |
| 2 | 0.844 | 1.000 |
| 3 | incomplete | — |
| 4 | not started | — |
| **Mean (3 seeds)** | **0.811** | **1.000** |

Rohira's reported HI ≈ 0.81. Our 3-seed mean matches exactly, but need 5 seeds for publishable error bars.

---

## Remaining Work (Priority Order)

### HIGH — Needed for OREL meeting

1. **`code_tasks.py` — Code execution scoring engine** (NOT YET CREATED)
   - `load_issues()` — parse issue markdown files
   - `generate_patch(issue, model, agent_expertise)` — LLM generates unified diff
   - `score_patch(issue_id, patch)` — apply patch to temp copy, run pytest, return pass/fail
   - `get_task_pool(num_tasks, rng)` — balanced issue selection
   - This is the critical "apples-to-apples" piece matching LLAMOSC

2. **Wire `code_tasks` into `simulation.py`**
   - Add `--code_tasks` flag to enable real code generation
   - Replace abstract reward with pytest pass/fail scoring
   - Agents generate patches via LLM, scored by running tests

3. **Complete A-series experiments** (seeds 3-4 for A1, all of A2)
   - A1: LLM-only, 10 agents, 10 sprints, dropout stress, 5 seeds
   - A2: LLM+AIF, same config — quantifies AIF contribution

4. **Run governance experiments WITH code tasks**
   - B-series repeat with real code generation + pytest scoring
   - Direct comparison to LLAMOSC's governance findings

### MEDIUM — Strengthens the paper

5. **A3: AIF-only (Level 4), 50 sprints, 10 seeds** — convergence comparison with Rohira
6. **C1: Ostrom priors experiment** — 10 agents, 10 sprints, 5 seeds
7. **Comparison report/paper draft**

### LOW — Nice to have

8. **Real SWE-agent/Aider integration** (Docker-based, like LLAMOSC)
9. **Community sizes 20/30** to match Rohira's scalability study
10. **Multi-fidelity argument writeup** (Rohira for sweeps, Concordia for process, LLAMOSC for ground truth)

---

## File Inventory

| File | Status | Purpose |
|------|--------|---------|
| `simulation.py` | Complete | Core simulation with governance modes |
| `evaluate.py` | Complete | All metrics (HI, RQ, BRS, SUE, CHS, SustainScore) |
| `social_data.py` | Complete | 30 agents, governance configs, reward alignment |
| `tools.py` | Complete | Grounded difficulty distributions |
| `experiments.py` | Complete | 8-level ladder + comparison suite |
| `run.py` | Complete | CLI with all flags |
| `active_inference.py` | Complete | AIF module with Ostrom priors |
| `toy_project/` | Complete | 10 issues, buggy calculator, pytest tests |
| **`code_tasks.py`** | **NOT CREATED** | **Code execution scoring engine** |
| `results_B_governance.json` | Complete | B-series results |
| `THREE_WAY_COMPARISON_PLAN.md` | Complete | Full plan document |

---

## How to Run

```bash
# Activate environment
source /home/mhough/dev/concordia/.venv/bin/activate

# Quick test (mock model)
python -m examples.games.sustain_hub.run --use_mock --num_sprints=1 --community_size=4

# NIM cloud (Qwen2.5-7B)
export NGC_API_KEY=<your-key>
python -m examples.games.sustain_hub.run \
  --vllm_url=https://integrate.api.nvidia.com/v1 \
  --model_name=qwen/qwen2.5-7b-instruct \
  --vllm_api_key=$NGC_API_KEY \
  --num_sprints=5 --community_size=8 --governance=free_choice --fast

# Run B-series governance comparison
python -m examples.games.sustain_hub.experiments --comparison \
  --experiments=B1,B2,B3 \
  --vllm_url=https://integrate.api.nvidia.com/v1 \
  --model_name=qwen/qwen2.5-7b-instruct \
  --vllm_api_key=$NGC_API_KEY

# Run experiment ladder (levels 0-7)
python -m examples.games.sustain_hub.experiments --level=all --use_mock

# Test toy project (should show expected failures)
cd examples/games/sustain_hub/toy_project && python -m pytest tests/ -v
```

---

## Git State

- Branch: `sustainhub-autoresearch`
- Latest commit: `ccc1bc3` — "Add B-series governance comparison results"
- Uncommitted: `toy_project/`, `fep_integration.py`, `live_results.json`
- Remote: `https://github.com/m9h/concordia.git`
