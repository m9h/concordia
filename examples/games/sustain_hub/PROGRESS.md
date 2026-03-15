# SustainHub Three-Way Comparison — Progress Report

**Date**: 2026-03-15 (updated)
**Target**: OREL meeting week of March 16-20
**Branch**: `sustainhub-autoresearch` @ `3be7fcc`

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

### Step 4: Code Tasks (LLAMOSC Comparison)
- `toy_project/` — 10 issues, buggy calculator, 36 pytest tests (30 fail on unpatched)
- `code_tasks.py` — LLM→patch→pytest scoring engine (load_issues, generate_patch, score_patch)
- `--code_tasks` flag wired into simulation.py and run.py
- Falls back to stochastic on failure; abstract mode unchanged

### Step 5: External Baseline Setup
- **Rohira SustainHub** cloned at `~/dev/rohira-sustainhub` with uv venv
  - `run_comparison.py` for headless batch runs
  - Results: HI=0.829±0.077 (baseline), 0.737 (15 agents), 0.733 (20), 0.683 (30)
- **LLAMOSC** cloned at `~/dev/llamosc` with uv venv
  - `nim_adapter.py` replaces hardcoded Ollama with NIM cloud support
  - `run_comparison.py` for headless batch runs
- **`run_three_way_comparison.py`** orchestrates all three systems

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

### External Baselines (Rohira RL, LLAMOSC LLM)

| System | Config | HI / HI_eq | Seeds |
|--------|--------|-------------|-------|
| Rohira | 10 agents, 10 steps | **0.829±0.077** | 5 |
| Rohira | 15 agents, 7 steps (canonical) | 0.737±0.023 | 5 |
| Rohira | 20 agents | 0.733±0.035 | 5 |
| Rohira | 30 agents | 0.683±0.047 | 5 |
| LLAMOSC | Auth, 5 contrib | 0.578±0.064 | 5 |
| LLAMOSC | Decentral, 5 contrib | 0.595±0.077 | 5 |
| LLAMOSC | Auth, 8 contrib, 10 issues | 0.518±0.044 | 5 |
| LLAMOSC | Decentral, 8 contrib, 10 issues | 0.519±0.040 | 5 |
| LLAMOSC | Auth, 10 contrib, 10 issues | 0.488±0.034 | 5 |
| LLAMOSC | Decentral, 10 contrib, 10 issues | 0.488±0.038 | 5 |

---

## Remaining Work (Priority Order)

### HIGH — Needed for OREL meeting

1. **Run code_tasks experiments on NIM** (B-series with `--code_tasks`)
   - Free choice, dictator, meritocratic with real pytest scoring
   - Direct comparison to LLAMOSC's governance + code findings

2. **Complete A-series on NIM** (NIM context issues — may need smaller config)
   - A1: seeds 3-4 (or re-run all 5 with 8 agents, 5 sprints)
   - A2: LLM+AIF, same config — quantifies AIF contribution

3. **Format final comparison table for OREL**

### MEDIUM — Strengthens the paper

4. **Rohira dropout stress results** (running)
5. **A3: AIF-only (Level 4), 50 sprints, 10 seeds** — convergence comparison
6. **C1: Ostrom priors experiment** — 10 agents, 10 sprints, 5 seeds
7. **Comparison report/paper draft**

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
