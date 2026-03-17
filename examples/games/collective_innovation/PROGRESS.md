# Collective Innovation — Progress Report

**Date**: 2026-03-16 (updated)
**Branch**: `sustainhub-autoresearch` on `m9h/concordia`
**Paper**: [Collective Innovation in Groups of Large Language Models](https://arxiv.org/abs/2407.05377) (Nisioti et al., 2024)

---

## Overview

Reimplementation of the experimental paradigm from arXiv:2407.05377 as a Concordia simulation. LLM agents play Little Alchemy 2: starting with four basic elements (water, fire, earth, air), they propose pairwise combinations each step to discover new elements from a recipe book of ~720 elements.

### Key Experimental Variables

| Variable | Options | Paper Finding |
|----------|---------|---------------|
| **Network connectivity** | `fully_connected`, `dynamic`, `isolated` | Dynamic (small groups + inter-group visits) outperforms both extremes |
| **Prompt structure** | `openended_single`, `openended_multi`, `targeted_single`, `targeted_multi` | Multi-agent context + targeted prompts yield most discoveries |

### Simulation Structure

Each step has two scenes:
1. **Sharing scene** — agents in the same network group share recent discoveries via free-form conversation
2. **Combination scene** — each agent chooses two elements from their inventory to combine; payoff engine checks the recipe book

### Metrics

- **Unique Discoveries**: total distinct elements discovered across all agents
- **Discovery Rate**: discoveries per step
- **Knowledge Diversity**: how evenly distributed knowledge is across agents (entropy-based)
- **Innovation Score**: composite metric

---

## Completed

### Initial Implementation (commit `006a5ee`)
- Full simulation with RecipeBook (720 elements from `alchemy_data.json`)
- Network connectivity modes: fully_connected, dynamic (stochastic inter-group visits), isolated
- Four prompt variants (open-ended/targeted × single/multi-agent context)
- Agent inventory tracking with discovery history
- Per-agent and global discovery metrics
- Runner script with all backend support (Gemini, NIM, Groq, Together, vLLM, mock)

### Bug Fixes (commit `3fba01e`)
- **CHOICE output type**: Switched combination scene from FREE to CHOICE output type, generating all valid element pairs as selectable options — eliminates LLM parsing failures
- **Robust fallback parser**: Scans narrative text for known element names when structured parsing fails (handles "I want to try combining water and fire..." style responses)
- **Expanded prefix stripping**: Added "I'd like to combine", "I will combine", "Let's combine", "Let's try", "with" separator
- **NIM retry parameters**: Increased to 15 tries, 5s delay, 2x backoff for NIM cloud stability

---

## Current Status

### What Works
- Mock model runs complete successfully
- All three connectivity modes functional
- Recipe book loads and combination lookup works
- Results JSON serialization and HTML log generation
- Multiple LLM backends (Gemini, NIM, Groq, Together, vLLM)

### Known Issues / Not Yet Tested
- [ ] Full LLM run on NIM cloud (verify CHOICE parsing works end-to-end with real LLM)
- [ ] Multi-step runs (50 steps × 6 agents) — check for rate limit / timeout issues
- [ ] Dynamic connectivity mode (inter-group visit probability tuning)
- [ ] Comparison across connectivity modes (paper's main result)
- [ ] Comparison across prompt modes

---

## Experiment Plan

### Phase 1: Validation (verify implementation matches paper)
- [ ] Run `fully_connected` × `openended_multi` × 50 steps × 6 agents × 3 seeds
- [ ] Run `isolated` × `openended_multi` × 50 steps × 6 agents × 3 seeds
- [ ] Run `dynamic` × `openended_multi` × 50 steps × 6 agents × 3 seeds
- [ ] Compare discovery curves: dynamic > fully_connected > isolated?

### Phase 2: Prompt Mode Sweep
- [ ] All 4 prompt modes × `dynamic` connectivity × 3 seeds
- [ ] Verify: targeted_multi > openended_multi > targeted_single > openended_single?

### Phase 3: Model Comparison
- [ ] Qwen2.5-7B vs Llama-3.1-8B vs larger models
- [ ] Does model capability monotonically improve collective innovation?

### Phase 4: Cross-Game Analysis
- [ ] Compare with SustainHub: different social dilemma, same infrastructure
- [ ] Do agents that innovate well also cooperate well? (across-game personality transfer)

---

## How to Run

```bash
# Activate environment
cd ~/dev/concordia
source .venv/bin/activate

# Quick test (mock model, 5 steps)
python -m examples.games.collective_innovation.run --use_mock --num_steps=5 --num_agents=4

# NIM cloud (Qwen2.5-7B)
python -m examples.games.collective_innovation.run \
  --nvidia_nim \
  --model_name=qwen/qwen2.5-7b-instruct \
  --num_steps=50 --num_agents=6 \
  --connectivity=dynamic --prompt_mode=openended_multi

# DGX Spark (local vLLM)
python -m examples.games.collective_innovation.run \
  --vllm_url=http://192.168.108.72:8000/v1 \
  --model_name=Qwen/Qwen2.5-7B-Instruct \
  --num_steps=50 --num_agents=6

# Groq (fast, free)
python -m examples.games.collective_innovation.run \
  --groq --model_name=llama-3.3-70b-versatile \
  --num_steps=50 --num_agents=6
```

---

## File Inventory

| File | Purpose |
|------|---------|
| `simulation.py` | Core simulation: RecipeBook, InnovationPayoff, network management, scene configuration |
| `run.py` | CLI runner with all backend support |
| `social_data.py` | Agent profiles and prompt templates |
| `alchemy_data.json` | Little Alchemy 2 recipe database (~720 elements) |
