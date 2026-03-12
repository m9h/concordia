# SustainHub Research: From RL to Active Inference

## Overview

This document tracks the theoretical progression for SustainHub's agent
decision-making framework: from pure LLM prompting → RL augmentation →
full Active Inference. The goal is to ground multi-agent social simulation
in a principled Bayesian decision-making framework while preserving the
richness of LLM-driven natural language reasoning.

## The Progression

| Level | Mechanism | Agent decides by... | Learning |
|-------|-----------|--------------------|---------|
| **1. LLM-only** (current) | Prompt → text response | Natural language reasoning in context | Episodic memory in prompt |
| **2. RL-augmented** (Rohira 2025) | SARSA Q-tables | Q-value lookup + epsilon-greedy | TD learning on rewards |
| **3. Active Inference** (target) | Expected Free Energy minimization | Minimizing surprise + maximizing info gain | Dirichlet parameter updates |
| **4. Hybrid LLM + AIF** (vision) | AIF provides policy probs, LLM provides reasoning | Combined Bayesian-rational + socially-aware | Both habit learning + semantic memory |

## Key Mathematical Mapping

### RL → Active Inference (Smith et al. 2022)

| RL Concept | Active Inference Equivalent | In SustainHub |
|-----------|---------------------------|---------------|
| State s | Hidden state s (partially observable) | Project health × task urgency |
| Action a | Policy π | Task choice (bug_fix, feature, docs, review, mentor, skip) |
| Reward R(s,a) | Prior preference C = ln P(o) | Preference for high HI observations |
| Q-value Q(s,a) | Negative Expected Free Energy -G(π) | Combined pragmatic + epistemic value |
| Value function V(s) | Free energy F(s) | Surprise about project state |
| Exploration (ε-greedy) | Epistemic value (information gain) | "I should do docs because I'm uncertain about docs backlog" |
| Exploitation | Pragmatic value (reward-seeking) | "I should do bug_fix because project needs it" |
| TD error δ | Prediction error ε | Difference between expected and observed HI |
| Policy gradient | Policy as softmax over -G + ln E | σ(-G(π) + ln E(π)) |

### POMDP Structure for SustainHub

```
Hidden States (not directly observable):
  Factor 1: Project Health    = {healthy, stressed, declining}
  Factor 2: Task Urgency      = {balanced, bugs_critical, docs_neglected, review_backlog}

Observations (what agents perceive):
  Modality 1: Harmony Index   = {high, medium, low}
  Modality 2: Task Outcome    = {success, partial, failure}

Actions (policies):
  {bug_fix, feature, documentation, code_review, mentor, skip}

Generative Model:
  A: P(observation | health, urgency)     — likelihood
  B: P(next_state | current_state, action) — transitions
  C: ln P(preferred observations)          — preferences (replaces reward)
  D: P(initial states)                     — prior beliefs (Dirichlet)
  E: P(policy)                             — habits (bridges from Q-values)
```

### The Key Equation: Expected Free Energy

```
G(π) = -pragmatic_value - epistemic_value

pragmatic_value = E_Q[ln P(o')]           # Do I get what I want?
epistemic_value = -E_Q[H[P(o'|s')]]      # Do I learn something?

P(π) = σ(-G(π) + ln E(π))                # Policy as softmax
```

Agents naturally balance exploitation (pragmatic: choose tasks that
yield preferred outcomes) with exploration (epistemic: choose tasks that
resolve uncertainty about the project state).

## Key References

### Core Active Inference Theory

- **Smith, R., Friston, K.J., & Whyte, C.J. (2022)**. A Step-by-Step
  Tutorial on Active Inference and its Application to Empirical Data.
  *Journal of Mathematical Psychology*.
  - Tutorial scripts: https://github.com/rssmith33/Active-Inference-Tutorial-Scripts
  - MATLAB implementation of POMDP active inference with Dirichlet learning
  - Key files: `Step_by_Step_AI_Guide.m` (main tutorial),
    `EFE_Precision_Updating.m`, `EFE_learning_novelty_term.m`

### RxInfer.jl: Probabilistic Programming + LLM Integration

- **RxInfer.jl** — Reactive message-passing framework for Bayesian inference
  - LLM integration examples: https://examples.rxinfer.com/categories/experimental_examples/large_language_models/
  - Key pattern: **LLMPrior** and **LLMObservation** nodes that treat
    LLM outputs as probability distributions within variational message
    passing (VMP)
  - Approach: LLM generates structured priors (Normal distributions with
    mean/variance via JSON schema prompting), which are then refined by
    Bayesian inference
  - Relevant to SustainHub: could use LLM to generate informative priors
    about project state, then refine via active inference

### Alexander Shaw's Toy Implementations

- **CPNS Lab**: https://cpnslab.com/
- **VFE_2D_Pong**: https://github.com/alexandershaw4/VFE_2D_Pong
  - MATLAB active inference agent playing Pong
  - Key pattern: short rollouts to evaluate actions via Expected Free Energy
  - Files: `selectActionEFE2D_rollout.m` (policy evaluation),
    `generativeModel.m` (factorized probabilistic model)
- **VariationalFreeEnergyToyDrone**: https://github.com/alexandershaw4/VariationalFreeEnergyToyDrone
  - Active inference agent controlling a drone
  - Includes epistemic foraging variant (`drone_with_epistemic/`)
  - Files: `ActiveInferenceDroneAgent.m`, `ActiveInferenceDroneAgentLearn.m`
  - Key insight: learning version accumulates Dirichlet parameters across
    episodes, exactly analogous to SustainHub agents learning across sprints

### SustainHub Original Framework

- **Rohira (2025)** — SustainHub: SARSA-based RL for open-source
  community sustainability
  - Uses Q-tables with TD learning for task selection
  - Our bridge: Q-values → E-vector (habit prior in active inference)
  - The SARSA reward structure maps directly onto C-matrix preferences

### Karpathy's Autoresearch

- **autoresearch**: https://github.com/karpathy/autoresearch
  - Autonomous ML research via iterative code modification
  - Applied here as meta-optimization: an AI agent modifies the simulation
    design itself, measuring SustainScore as the evaluation metric

## Implementation Plan

### Phase 1: Current (LLM-only)
- [x] LLM agents with natural language decision-making
- [x] Episodic memory via Concordia's memory components
- [x] Harmony Index and Resilience Quotient metrics

### Phase 2: Active Inference Module
- [x] `active_inference.py` — standalone POMDP with A/B/C/D/E matrices
- [ ] Integration with Concordia agent components
- [ ] LLM + AIF hybrid: `format_aif_context_for_llm()` feeds AIF state
  into LLM prompts as "[Internal Assessment]" context
- [ ] Validate on simple scenarios (4 agents, 1 sprint)

### Phase 3: Learning Across Sprints
- [ ] Dirichlet parameter updates (D-matrix learning)
- [ ] Habit accumulation (E-vector learning across sprints)
- [ ] Compare learning curves: LLM-only vs AIF vs hybrid

### Phase 4: Scaling
- [ ] Local model backend (Qwen 2.5-7B via vLLM/SGLang on DGX Spark)
- [ ] 64-agent simulations with async engine
- [ ] 10+ sprint runs observing long-term adaptation
- [ ] Autoresearch loop optimizing AIF hyperparameters
  (gamma, alpha, learning_rate, prior shapes)

## Target Architecture: RxInfer-Style LLM + Message Passing

The RxInfer.jl team (ReactiveBayes/BIASlab) has the most principled
integration of LLMs with Bayesian inference. Their pattern:

### The Pattern: LLM as Probabilistic Node

Instead of LLMs *replacing* decision-making, LLMs participate as
**nodes in a factor graph** alongside standard probabilistic nodes:

```
Factor Graph for SustainHub Agent:

  [Agent Backstory]──→ LLMPrior ──→ ProjectHealthBelief
                                          │
  [Sprint Outcome Text]──→ LLMObservation ─┘
                                          │
                                    BayesianInference
                                          │
                                    PolicySelection ──→ Action
                                          │
                              ExpectedFreeEnergy ←── Preferences
```

### Two Key Node Types

**LLMPrior**: Given context (agent backstory, role, sprint history),
the LLM generates a *structured probability distribution* as a prior:

```python
# Pseudocode (Python equivalent of RxInfer's Julia pattern)
def llm_prior(context: str, task: str) -> NormalDistribution:
    """LLM generates informed prior as mean + variance."""
    prompt = f"""Given this context: {context}
    Task: {task}
    Respond with JSON: {{"analysis": "...", "mean": float, "variance": float}}"""
    response = model.sample_text(prompt)
    params = json.loads(response)
    return Normal(mean=params["mean"], variance=params["variance"])
```

**LLMObservation**: Given observed text (sprint outcome descriptions),
the LLM maps text to a latent numerical representation:

```python
def llm_observation(observed_text: str, task: str) -> NormalDistribution:
    """LLM maps text observation to latent state estimate."""
    prompt = f"""Observed: {observed_text}
    Task: {task}
    Rate this as a number with uncertainty.
    Respond with JSON: {{"analysis": "...", "mean": float, "variance": float}}"""
    response = model.sample_text(prompt)
    params = json.loads(response)
    return Normal(mean=params["mean"], variance=params["variance"])
```

### Why This Is Better Than Our Current Approach

| Aspect | Current (naive hybrid) | RxInfer pattern |
|--------|----------------------|----------------|
| LLM role | Generates text decisions directly | Generates *distributions* that feed inference |
| Uncertainty | Implicit (LLM temperature) | Explicit (variance parameter) |
| Composability | LLM is monolithic decision-maker | LLM is one node among many |
| Learning | Prompt-based (fragile) | Dirichlet parameter updates (principled) |
| Interpretability | "Why did it choose X?" → unclear | Factor graph shows information flow |

### Key RxInfer Examples Relevant to SustainHub

- **POMDP Control**: https://examples.rxinfer.com/categories/basic_examples/pomdp_control/
  - Basic active inference for partially observable environments
- **Active Inference Mountain Car**: https://examples.rxinfer.com/categories/advanced_examples/active_inference_mountain_car/
  - Full active inference with expected free energy minimization
- **Multi-Agent Trajectory Planning**: https://examples.rxinfer.com/categories/advanced_examples/multi-agent_trajectory_planning/
  - Multi-agent coordination via message passing — directly relevant
- **Hierarchical Gaussian Filter**: https://examples.rxinfer.com/categories/problem_specific/hierarchical_gaussian_filter/
  - Hierarchical learning of volatility — models changing environments
- **LLM Integration**: https://examples.rxinfer.com/categories/experimental_examples/large_language_models/
  - The LLMPrior/LLMObservation pattern described above

### Acknowledged Limitations (from RxInfer team)

1. **"Unprincipled" uncertainty**: LLMs don't genuinely quantify epistemic
   uncertainty — they pattern-match to training expressions of confidence
2. **Fixed functional forms**: Currently hardcoded to Normal distributions
3. **Message product handling**: Combining multiple incoming messages to
   LLM nodes is unsolved
4. **Alternatives**: Token log-probabilities could provide more grounded
   uncertainty estimates

### Implementation Path for SustainHub

**Phase 1** (current `active_inference.py`): Hardcoded A/B/C/D/E matrices,
no LLM in the inference loop. LLM only for natural language generation.

**Phase 2**: Replace hardcoded A-matrix with LLMObservation node.
The LLM reads sprint outcomes and generates likelihood estimates.
Still use hardcoded B/C matrices.

**Phase 3**: Replace D-matrix priors with LLMPrior node.
The LLM reads agent backstory and generates informed initial beliefs
about project health. Dirichlet learning refines these across sprints.

**Phase 4**: Full factor graph with message passing. Multiple LLM nodes
feeding into variational inference. This is the "RxInfer in Python" vision.

## Other Relevant Resources

- **RxInfer.jl source**: https://github.com/ReactiveBayes/RxInfer.jl
- **RxInfer examples repo**: https://github.com/ReactiveBayes/RxInferExamples.jl
- **BIASlab (Bert de Vries' group)**: https://biaslab.github.io/
- **CPNS Lab (Alexander Shaw)**: https://cpnslab.com/

## Open Questions

1. **How to combine LLM and AIF signals?** The RxInfer pattern answers
   this: LLMs generate *distributions*, not decisions. Bayesian inference
   combines them. But translating this from Julia factor graphs to
   Python/Concordia requires building a minimal message passing framework.

2. **Precision (gamma) dynamics**: Should precision increase over sprints
   as agents become more confident? This would model the transition from
   exploration (early sprints) to exploitation (later sprints).

3. **Multi-agent active inference**: Each agent has its own generative
   model. Do agents learn to model *each other's* beliefs? (Theory of
   mind via nested active inference.) RxInfer's multi-agent trajectory
   planning example may provide patterns here.

4. **Stress as precision collapse**: Stress events could be modeled as
   sudden drops in precision (gamma → 0), making agents more exploratory
   and less predictable. This connects to the neuroscience of stress
   (reduced prefrontal precision → impulsive behavior).

5. **Can we use RxInfer.jl directly?** Julia ↔ Python interop via
   PyJulia or JuliaCall is possible but adds complexity. Alternative:
   port the essential message passing logic to Python (it's not much code
   for the POMDP case).
