# Autoresearch for Concordia: Research Dimensions & Implementation Plan

**Date**: 2026-03-13
**Branch**: `sustainhub-autoresearch`
**Repo**: `m9h/concordia`

---

## Context

We've successfully cloned the Concordia fork (`m9h/concordia`, branch `sustainhub-autoresearch`), deployed vLLM (Qwen2.5-7B-Instruct) on the DGX Spark at `192.168.108.72:8000`, and completed an end-to-end SustainHub simulation. The infrastructure works. Now: what research should this infrastructure *do*?

The Karpathy autoresearch loop (hypothesis → code modification → experiment → analysis → iterate) is the meta-frame. But Concordia is uniquely positioned because the LLM is simultaneously the **experimental instrument** (running simulations) and the **object of study** (its behavior *is* the data). This duality creates research dimensions that don't exist in standard ML autoresearch.

---

## Research Dimensions Where LLMs, Agent Harnesses, and Autoresearch Intersect

### Dimension 1: LLM as Cognitive Engine vs. Object of Study

In Concordia, the LLM plays two roles that can be studied independently:

- **Tool role**: The LLM generates natural language reasoning, backstories, dialogue. Here it's infrastructure — like a physics engine in a game.
- **Object role**: The LLM's outputs *are* the experimental data. "Did 7B agents discover cooperation?" is a question about model capability, not simulation design.

**Autoresearch opportunity**: The `model_sweep.sh` already treats the LLM backbone as an independent variable. A true autoresearch loop would go further — automatically discovering which *prompting strategies* elicit cooperation from which model families. The hypothesis space is: `{model × prompt_template × reward_structure → cooperation_metric}`.

### Dimension 2: The Active Inference Bridge (Structured Reasoning ↔ Language Reasoning)

The 8-level experiment ladder (experiments.py) is the key innovation. Levels 0-6 are pure math (no LLM), Level 7 is the hybrid. This creates a unique autoresearch target:

- **Levels 0-6** run in seconds (CPU-only). An autoresearch loop can sweep A/B/C/D/E matrix parameters at thousands of configs/hour.
- **Level 7** (LLM-as-Node) validates whether the fast-layer's optimal parameters also work when LLMs provide the reasoning. If they diverge, *that divergence is a research finding*.
- The `format_aif_context_for_llm()` function is the seam — it injects Bayesian state into LLM prompts as "[Internal Assessment]". Autoresearch can vary *how much* of the AIF state the LLM sees and measure the effect.

### Dimension 3: Emergent Social Phenomena as Fitness Function

Standard autoresearch optimizes a scalar metric (loss, accuracy). SustainHub's `SustainScore` is a composite that explicitly rewards *interesting* dynamics, not just performance:

```
SustainScore = HI × (1 + RQ) × Fairness × StrategyDiversity × StressValidity
```

This means the autoresearch loop optimizes for *scientifically valuable simulations* — ones where agents exhibit real social dilemma dynamics (free-riding, coalition formation, burnout, graduated sanctions). A simulation where everyone cooperates trivially scores low on `StressValidity`.

### Dimension 4: Multi-Fidelity Optimization (System 1 / System 2)

The dual-process architecture in RESEARCH.md maps directly to autoresearch tiers:

| Tier | System | Speed | What it optimizes |
|------|--------|-------|-------------------|
| Fast inner loop | AIF Levels 0-6 (CPU/GPU) | 10³-10⁶ steps/sec | A/B/C/D/E matrices, gamma, alpha |
| Slow validation | Concordia + LLM (DGX Spark) | ~0.1 steps/sec | Prompt templates, conversation dynamics |
| Phase transition trigger | Critical Slowing Down detection | Real-time | When to invoke System 2 deliberation |

Autoresearch operates at *all three tiers simultaneously*. The fast loop discovers parameter regimes; the slow loop validates them with rich language; the trigger logic learns *when* language-based reasoning matters.

### Dimension 5: The Ostrom-Bayesian Hypothesis

The deepest research question: **Can Active Inference explain why Ostrom's Design Principles work?**

Ostrom's 8 principles for governing the commons (clear boundaries, graduated sanctions, collective-choice arenas, etc.) are empirical findings from fieldwork. The SustainHub hypothesis reframes them as Bayesian priors that minimize social free energy. If an autoresearch loop discovers that agents with Ostrom-aligned priors (D-matrix) consistently outperform agents without them, that's a computational proof of Ostrom's theory.

This is the kind of research question that *only* an autoresearch loop can answer — the parameter space (8 principles × prior strengths × agent counts × stress conditions) is too large for manual experimentation.

---

## Implementation Plan

### Step 1: Fix Local Environment & Verify DGX Spark Connection

**Files**: `concordia/contrib/language_models/vllm_remote.py`, `examples/games/sustain_hub/run.py`

- Resolve the stashed local changes (our `gemini_model` import fix vs. the pulled version)
- Verify the upstream `vllm_remote.py` (commit `7e1f270`) handles our `sample_choice` + timeout needs
- Run a quick smoke test: `--use_mock --num_sprints=1 --community_size=4`
- Run a DGX test: `--vllm_url=http://192.168.108.72:8000/v1 --num_sprints=1 --community_size=4 --fast --skip_backstory`

### Step 2: Run the Full Experiment Ladder (No LLM, Instant)

**Files**: `examples/games/sustain_hub/experiments.py`, `examples/games/sustain_hub/evaluate.py`

- Run all 8 levels with 10 seeds: `python -m examples.games.sustain_hub.experiments --level=all --num_sprints=5`
- This produces the baseline data showing what each AIF concept contributes *without* any LLM involvement
- Compute SustainScore per level → establishes the "math-only" frontier

### Step 3: Run Concordia Simulations with vLLM on DGX Spark

**Files**: `bin/overnight_runner.py`, `examples/games/sustain_hub/run.py`

- Scale up the vLLM server: increase `gpu_memory_utilization` from 0.3 → 0.85 (3-4x throughput)
- Run overnight batch: 5 seeds × 8 agents × 3 sprints with Active Inference enabled
- Compare Level 7 (LLM+AIF) results against Level 4-6 (pure AIF) from Step 2
- **Key measurement**: Does the LLM improve or degrade cooperation vs. pure Bayesian agents?

### Step 4: Implement the Autoresearch Loop

**New file**: `bin/autoresearch.py`
**Modifies**: `examples/games/sustain_hub/simulation.py`, `social_data.py`, `scenario_config.py`

Following `program.md`, implement the Karpathy-style loop:

1. **Git-based experiment tracking**: Each hypothesis = one commit on an autoresearch branch
2. **Automated SustainScore evaluation**: Run Tier 1 (4 agents, 1 sprint, ~2 min), compute SustainScore
3. **Keep/revert logic**: If SustainScore improves → keep; if worse → `git reset --hard HEAD~1`
4. **Variation axes** (ordered by expected impact):
   - Layer 1: Prompt engineering (`CALL_TO_SPEECH`, `DECISION_PREMISE`)
   - Layer 2: Reward shaping (preferred:non-preferred ratio, coverage bonuses)
   - Layer 3: Simulation mechanics (HI formula, information design)
   - Layer 4: Agent design (backstory wording, role composition)
5. **Multi-fidelity**: Fast iteration on Gemini Flash Lite or local vLLM, validate winners at full scale

### Step 5: Model Sweep (LLM as Object of Study)

**Files**: `bin/model_sweep.sh`, `bin/overnight_runner.py`

Using the DGX Spark's 128GB unified memory:
- Tier 1 (7-8B): Llama-3.1-8B, Qwen-2.5-7B, DeepSeek-R1-7B — 5 seeds each
- Tier 2 (70B): Llama-3.1-70B, Qwen-2.5-72B, DeepSeek-R1-70B — 3 seeds each

Research question: *"Does model capability monotonically improve cooperation, or is there a threshold/saturation?"*
Secondary: *"Do RL-trained models (DeepSeek-R1) cooperate differently than supervised (Llama)?"*

### Step 6: Save Research Configuration as Memory

Save the DGX Spark setup, vLLM configuration, and research directions as project memories for continuity across sessions.

---

## Verification Checklist

1. **Smoke test**: `python -m examples.games.sustain_hub.run --use_mock --num_sprints=1 --community_size=4` completes without error
2. **DGX connectivity**: `curl http://192.168.108.72:8000/v1/models` returns the model
3. **Experiment ladder**: `python -m examples.games.sustain_hub.experiments --level=0,1,2 --num_sprints=2` produces comparison table
4. **Full sim on DGX**: `python -m examples.games.sustain_hub.run --vllm_url=... --model_name=Qwen/Qwen2.5-7B-Instruct --num_sprints=2 --fast --community_size=4` completes with results.json
5. **Autoresearch loop**: Completes at least 3 iterations (hypothesis → experiment → keep/revert)

---

## Key Files Reference

| File | Role | Lines |
|------|------|-------|
| `examples/games/sustain_hub/simulation.py` | Core simulation engine | 1361 |
| `examples/games/sustain_hub/active_inference.py` | POMDP + A/B/C/D/E matrices | 639 |
| `examples/games/sustain_hub/experiments.py` | 8-level experiment ladder | 700 |
| `examples/games/sustain_hub/evaluate.py` | SustainScore computation | 235 |
| `examples/games/sustain_hub/social_data.py` | Agent profiles, rewards, prompts | 438 |
| `examples/games/sustain_hub/tools.py` | Agent tools (AutoCodeRover, Stats, Mentorship) | 102 |
| `examples/games/sustain_hub/scenario_config.py` | Sprint/scenario configuration | — |
| `examples/games/sustain_hub/program.md` | Autoresearch protocol | 231 |
| `examples/games/sustain_hub/RESEARCH.md` | Theoretical framework | 531 |
| `concordia/contrib/language_models/vllm_remote.py` | DGX Spark vLLM adapter | 117 |
| `bin/overnight_runner.py` | Batch orchestration | 283 |
| `bin/model_sweep.sh` | LLM backbone comparison | 98 |
| `bin/setup_vllm.sh` | vLLM server setup for DGX Spark | 94 |
| `concordia/document/interactive_document.py` | Prompt construction (LLM-as-cognitive-engine seam) | — |

---

## Infrastructure

- **DGX Spark**: `192.168.108.72`, 128GB unified memory, Grace Blackwell GB10
- **vLLM server**: port 8000, OpenAI-compatible API
- **Current model**: `Qwen/Qwen2.5-7B-Instruct`
- **Python**: >=3.12, managed with `uv`
- **Embeddings**: `sentence-transformers` (local, CPU)

---

## SustainScore Formula

```
SustainScore = HI × (1 + RQ) × Fairness × StrategyDiversity × StressValidity
```

| Component | Range | What it measures |
|-----------|-------|-----------------|
| HI (Harmony Index) | 0-1 | Community sustainability |
| RQ (Resilience Quotient) | 0-1 | Recovery from stress events |
| Fairness | 0-1 | 1 - Gini coefficient of agent scores |
| StrategyDiversity | 0-1 | Fraction of agents changing task type |
| StressValidity | 0.5 or 1.0 | Whether stress events actually affect outcomes |

**Targets**: >0.60 good, >0.80 excellent

---

## Experiment Ladder Summary

| Level | Concept Added | Cumulative Capabilities |
|-------|---------------|------------------------|
| 0 | Pure RL baseline | Reward signals only |
| 1 | +Prediction Error | Surprise tracking |
| 2 | +Epistemic Value | Information gain in decisions |
| 3 | +Belief Updating | Variational inference over hidden states |
| 4 | +Expected Free Energy | EFE-based policy selection |
| 5 | +Habit Learning | Dirichlet parameter accumulation |
| 6 | +Precision Dynamics | Adaptive exploration/exploitation |
| 7 | +LLM-as-Node | LLM as probabilistic inference node in factor graph |
