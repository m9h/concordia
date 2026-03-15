# SustainHub Architecture & Integration Guide

## Overview

SustainHub is a Concordia-based multi-agent simulation of open-source community dynamics. LLM agents with distinct roles (Contributors, Innovators, Knowledge Curators, Maintainers) manage a shared task queue across sprints, facing social dilemmas around task allocation, mentoring, and project sustainability.

The system uniquely combines:
- **LLM cognition** (natural language reasoning, social interaction)
- **Active Inference** (Bayesian decision-making with Expected Free Energy)
- **Ostrom governance priors** (formalized commons governance principles)
- **Grounded code execution** (pytest-scored patches against a toy project)

## File Map

```
examples/games/sustain_hub/
├── simulation.py          # Core simulation engine (scenes, agents, payoff)
├── social_data.py         # Agent profiles, roles, rewards, governance configs
├── active_inference.py    # Active Inference module (POMDP, EFE, Ostrom priors)
├── evaluate.py            # Metrics (HI, RQ, BRS, SUE, CHS, SustainScore)
├── experiments.py         # 8-level experiment ladder + comparison suite
├── run.py                 # CLI runner with all flags
├── tools.py               # Agent tools (AutoCodeRover, grounded difficulty)
├── code_tasks.py          # Code execution scoring engine (LLM→patch→pytest)
├── fep_integration.py     # Free Energy Principle integration (experimental)
├── toy_project/           # Toy calculator project for code tasks
│   ├── calculator/
│   │   ├── __init__.py
│   │   └── calculator.py  # Buggy calculator (5 bugs, 5 missing features)
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_calculator.py  # 10 test classes, 36 tests total
│   ├── issues/
│   │   ├── issue_1.md     # Fix: division by zero (bug_fix, easy)
│   │   ├── issue_2.md     # Fix: negative sqrt (bug_fix, easy)
│   │   ├── issue_3.md     # Fix: factorial validation (bug_fix, medium)
│   │   ├── issue_4.md     # Feature: percentage method (feature, easy)
│   │   ├── issue_5.md     # Feature: absolute value (feature, easy)
│   │   ├── issue_6.md     # Feature: logarithm (feature, medium)
│   │   ├── issue_7.md     # Feature: mean (feature, medium)
│   │   ├── issue_8.md     # Feature: is_prime (feature, hard)
│   │   ├── issue_9.md     # Docs: docstrings (documentation, easy)
│   │   └── issue_10.md    # Fix: history records errors (bug_fix, hard)
│   └── conftest.py        # pytest path configuration
├── PROGRESS.md            # Current progress and results
├── ARCHITECTURE.md        # This file
├── THREE_WAY_COMPARISON_PLAN.md  # Full comparison plan
└── results_B_governance.json     # B-series experiment results
```

## Three-Way Comparison

| System | Origin | Agent Cognition | Task Fidelity | Governance |
|--------|--------|----------------|---------------|------------|
| **Rohira SustainHub** | OREL GSoC '24 | MAB (Thompson Sampling) + SARSA | Abstract (task types) | Fixed allocation |
| **Basta LLAMOSC** | OREL GSoC '24 | LLM + AutoCodeRover | Real code (Docker) | Dictator / Meritocratic |
| **Concordia SustainHub** | This fork | LLM + Active Inference hybrid | Both abstract & grounded | Scene-based deliberation |

### Matched Metrics

| Metric | Formula | All 3 Systems? |
|--------|---------|----------------|
| HI (Harmony Index) | `mean(individual_HI)` | Yes |
| RQ (Resilience Quotient) | HI recovery after stress | Yes |
| BRS (Burnout Risk Score) | `consecutive_nonpreferred / total_sprints` | Concordia + Rohira |
| SUE (Skill Utilization) | `1.0 preferred, 0.5 adjacent, 0.0 unrelated` | Concordia + Rohira |
| CHS (Community Health) | `0.3*HI + 0.25*(1-BRS) + 0.25*SUE + 0.2*RQ` | Concordia + Rohira |

## Simulation Flow

```
Sprint N:
  1. [GM] Generate stress event (if enabled)
  2. [GM] Present task queue to agents
  3. [Agents] Conversation scene (unless --fast)
  4. [Agents] Task selection (governance-dependent):
     - free_choice: agents self-select
     - dictator: GM assigns by role
     - meritocratic: priority by track record
  5. [GM] Resolve actions → compute rewards
  6. [Evaluate] Compute HI, BRS, SUE, CHS for sprint
  7. [AIF] Update Active Inference beliefs (if enabled)
```

## Code Task Pipeline (LLAMOSC Comparison)

```
1. load_issues()          → 10 issues from toy_project/issues/
2. get_task_pool(N, rng)  → balanced selection of N issues
3. generate_patch(issue, model, expertise)  → LLM writes unified diff
4. score_patch(issue_id, patch)  → copy project, apply patch, run pytest
5. score_code_task(issue_id, patch)  → map pass rate to SustainHub rewards

LLAMOSC equivalent:
  load_issues → parse markdown in calculator_project/issues/
  generate_patch → AutoCodeRover in Docker
  score_patch → pytest in Docker
  score → experience/motivation/code_quality metrics
```

## Experiment Ladder

| Level | Description | Agent Type | Model Needed |
|-------|------------|-----------|-------------|
| 0 | Pure RL baseline | Reward-only | None |
| 1 | + Prediction Error | AIF (surprise) | None |
| 2 | + Epistemic Value | AIF (info gain) | None |
| 3 | + Belief Updating | AIF (VFE) | None |
| 4 | + Expected Free Energy | AIF (EFE) | None |
| 5 | + Habit Learning | AIF (Dirichlet) | None |
| 6 | + Ostrom Governance | AIF + Ostrom | None |
| 7 | + LLM Cognition | LLM + AIF | Qwen2.5-7B+ |

## Running Experiments

```bash
# Activate environment
source /home/mhough/dev/concordia/.venv/bin/activate

# Quick test (no LLM needed)
python -m examples.games.sustain_hub.experiments --level=0,1,2,3,4

# Full ladder with mock LLM
python -m examples.games.sustain_hub.experiments --level=all --use_mock

# NIM cloud experiments
export NGC_API_KEY=<key>
python -m examples.games.sustain_hub.experiments --comparison \
  --experiments=B1,B2,B3 \
  --vllm_url=https://integrate.api.nvidia.com/v1 \
  --model_name=qwen/qwen2.5-7b-instruct \
  --vllm_api_key=$NGC_API_KEY

# Single run with governance
python -m examples.games.sustain_hub.run \
  --vllm_url=https://integrate.api.nvidia.com/v1 \
  --model_name=qwen/qwen2.5-7b-instruct \
  --vllm_api_key=$NGC_API_KEY \
  --num_sprints=5 --community_size=8 --governance=meritocratic --fast

# Verify toy project (all issues should fail)
cd examples/games/sustain_hub/toy_project && python -m pytest tests/ -v

# Test code_tasks module
python -m examples.games.sustain_hub.code_tasks
```

## Key Design Decisions

1. **Abstract + Grounded tasks**: The simulation supports both abstract task types (stochastic success based on expertise) and grounded code tasks (pytest-scored). The `--code_tasks` flag switches modes. Abstract mode is cheaper and faster for parameter sweeps; grounded mode matches LLAMOSC for direct comparison.

2. **Active Inference as Bayesian layer**: AIF doesn't replace LLM reasoning — it provides structured priors and belief updates that *inform* the LLM prompt. The "[Internal Assessment]" block gives the LLM access to Bayesian posteriors without constraining its output.

3. **Governance via scene structure**: Rather than hardcoding governance as a reward modifier, governance modes change the *scene structure* — dictator mode adds a GM assignment scene, meritocratic mode sorts the action space by track record. The social dynamics emerge naturally.

4. **Ostrom priors as Bayesian formalization**: Ostrom's 8 design principles for commons governance are encoded as prior distributions over policy parameters. This connects institutional economics to Active Inference theory — a novel theoretical contribution.

5. **Multi-fidelity design**: Levels 0-6 run without any LLM (fast, deterministic, cheap). Level 7 adds LLM cognition. This allows parameter sweeps on cheap levels, then validation on expensive LLM runs.
