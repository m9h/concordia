# SustainHub TODO — OREL Meeting Prep (Week of March 16-20, 2026)

## Critical Path (Must Have)

### 1. Wire `code_tasks` into `simulation.py`
- [x] Add `--code_tasks` flag to `run.py`
- [x] In `simulation.py`, when `code_tasks=True`:
  - Call `code_tasks.get_task_pool()` to replace abstract task queue
  - After agent selects a task, call `code_tasks.generate_patch()` to have LLM write a fix
  - Call `code_tasks.score_code_task()` for pytest-based reward instead of stochastic
- [x] Test with mock model first, then NIM cloud
- **Why**: This is the "apples-to-apples" LLAMOSC comparison piece
- **Status**: DONE (wired in PROGRESS.md Step 4)

### 2. Complete A-series experiments (Rohira comparison)
- [x] A1: 3/5 seeds complete (HI mean = 0.811 from 3 seeds)
- [ ] A1: Seeds 3-4 (need to rerun on NIM)
- [ ] A2: LLM+AIF, 10 agents, 10 sprints, dropout stress, 5 seeds
- **Why**: Direct comparison to Rohira's reported HI ≈ 0.81

### 3. Run governance experiments WITH code tasks
- [ ] B1-code: Free choice + code tasks
- [ ] B2-code: Dictator + code tasks
- [ ] B3-code: Meritocratic + code tasks
- **Why**: Matches LLAMOSC's governance + real code experiment exactly

### 4. Write comparison table
- [ ] Format: System | Governance | HI | RQ | BRS | SUE | CHS | Code Score
- [ ] Include Rohira's published numbers
- [ ] Include LLAMOSC's published numbers
- [ ] Include our numbers (abstract and grounded)
- **Why**: This is the deliverable for the OREL meeting

## Important (Should Have)

### 5. A3: AIF-only convergence comparison
- [ ] Level 4 (EFE), 10 agents, 50 sprints, 10 seeds
- [ ] Plot convergence curve alongside Rohira's SARSA convergence
- **Why**: Shows our math matches Rohira's when using the same (non-LLM) approach

### 6. C1: Ostrom priors experiment
- [ ] 10 agents, 10 sprints, 5 seeds
- [ ] Compare with/without Ostrom priors (A2 vs C1)
- **Why**: Novel contribution — Bayesian formalization of Ostrom's design principles

### 7. Autoresearch Layer 5 (governance)
- [ ] Run autoresearch with governance dimension enabled
- [ ] Collect results across prompt × governance × model variations
- **Why**: Automated optimization over the governance axis

## Nice to Have

### 8. Multi-model comparison
- [ ] Test with different SLMs: Qwen2.5-7B, Phi-3, Gemma-2-9B
- [ ] Compare LLM capability vs. governance quality

### 9. Community scaling
- [ ] Run at sizes 10, 20, 30 to match Rohira's scalability study
- [ ] 30-agent profiles already exist in social_data.py

### 10. SWE-agent integration
- [ ] Replace LLM patch generation with full SWE-agent in Docker
- [ ] Closer match to LLAMOSC's AutoCodeRover approach

---

## Recently Completed (Phase 1-2 Community Dynamics)

- [x] **1.1 Close AIF Loop**: `observe()` + `learn()` called after each sprint (was open-loop)
- [x] **1.2 Belief Alignment Index**: JSD-based BAI metric in `evaluate.py`
- [x] **1.3 Dialogue Acts**: 8-category classification + Coordination/Social Pressure indices
- [x] **2.1 Dynamic Trust Network**: Bayesian trust updating, replaces static relational matrix
- [x] **2.2 Norm Emergence Tracking**: 5 norms, compliance/strength/emergence metrics
- [x] **2.3 Burnout Cascade Modeling**: per-agent burnout with contagion through trust network
- [x] **2.4 Coalition Detection**: belief clustering per sprint, stable coalition identification
- [x] **DGX Spark vLLM fix**: `NVIDIA_DISABLE_REQUIRE=1` for driver 580 compat

## Current Results Summary

### B-Series: Governance (NIM Cloud, Qwen2.5-7B, 5 seeds × 5 sprints × 8 agents)

| Mode | HI | BRS | SUE | CHS |
|------|-----|------|------|------|
| Free choice | **0.831±0.035** | 0.030 | 0.970 | **0.934** |
| Meritocratic | 0.811±0.018 | 0.055 | 0.957 | 0.919 |
| Dictator | 0.806±0.046 | 0.080 | 0.943 | 0.907 |

### A-Series: Rohira Match (Partial — 3/5 seeds, 10 agents, 10 sprints)

| Seed | HI | RQ |
|------|------|------|
| 0 | 0.847 | 1.000 |
| 1 | 0.742 | 1.000 |
| 2 | 0.844 | 1.000 |
| **Mean** | **0.811** | **1.000** |

Rohira's reported HI ≈ 0.81 — our 3-seed mean matches.

### Toy Project Verification

- 10 issues (4 bug_fix, 5 feature, 1 documentation)
- 36 pytest tests total
- Unpatched: 30 FAIL, 6 PASS (correct — all issues need fixing)
- `code_tasks.py` loads all issues and generates balanced task pools
