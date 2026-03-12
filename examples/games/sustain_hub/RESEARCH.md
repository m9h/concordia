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

## Experiment Ladder

The experiment ladder (`experiments.py`) progressively adds active inference
concepts to a pure RL baseline. Each level introduces exactly one new concept
so we can measure its individual contribution.

```
python -m examples.games.sustain_hub.experiments --level=all --num_sprints=5
```

### Level 0: Pure RL Baseline
- **What**: Reward-driven task selection (preferred = +3, other = +1)
- **RL analog**: Q-learning / SARSA with fixed policy
- **AIF mechanism**: None
- **Expected behavior**: Agents always pick preferred tasks (greedy)

### Level 1: + Prediction Error
- **What**: Track surprise = -log P(observation | beliefs)
- **RL analog**: TD error δ = r + γV(s') - V(s)
- **AIF mechanism**: Free energy F measures model-world mismatch
- **Expected behavior**: Agents notice when outcomes don't match expectations

### Level 2: + Epistemic Value
- **What**: Actions that reduce uncertainty get a bonus
- **RL analog**: ε-greedy → UCB (principled exploration)
- **AIF mechanism**: Epistemic value = expected information gain
- **Expected behavior**: Agents occasionally explore non-preferred tasks

### Level 3: + Belief Updating
- **What**: Maintain and update beliefs about hidden project state
- **RL analog**: State estimation (Kalman filter)
- **AIF mechanism**: Q(s) ∝ P(o|s) × P(s) via variational message passing
- **Expected behavior**: Agents form accurate models of project health

### Level 4: + Expected Free Energy Policy
- **What**: Single objective combining reward + info gain
- **RL analog**: Q-value → negative EFE
- **AIF mechanism**: G(π) = -pragmatic - epistemic; P(π) = σ(-G + ln E)
- **Expected behavior**: Balanced exploitation/exploration without ad-hoc tuning

### Level 5: + Habit Learning
- **What**: Dirichlet concentration parameters accumulate across sprints
- **RL analog**: Q-value accumulation across episodes
- **AIF mechanism**: E(a) += η × outcome_valence
- **Expected behavior**: Agents develop persistent preferences that evolve

### Level 6: + Precision Dynamics
- **What**: γ (confidence) adapts based on prediction error history
- **RL analog**: Temperature annealing in softmax policy
- **AIF mechanism**: Low PE → high γ → exploit; High PE → low γ → explore
- **Expected behavior**: Agents become more decisive as they learn, but
  stress events trigger re-exploration

### Level 7: + LLM-as-Node (Full Hybrid)
- **What**: LLM generates probability distributions that feed into
  Bayesian inference (RxInfer pattern)
- **RL analog**: No RL analog — this is beyond RL
- **AIF mechanism**: LLM → distribution → factor graph → VMP → policy
- **Expected behavior**: Rich semantic understanding + principled inference

### What to Look For

As levels increase, we expect to see:
1. **Coverage increases**: Agents stop always picking preferred tasks
2. **Strategy diversity increases**: Agents change behavior across sprints
3. **HI stabilizes**: Less variance in Harmony Index across runs
4. **Stress recovery improves**: Agents adapt faster after disruptions
5. **Emergent specialization**: Agents naturally divide labor based on beliefs

The comparison table at the end shows how each concept contributes.

## Implementation Plan

### Phase 1: Standalone AIF (current)
- [x] `active_inference.py` — POMDP with A/B/C/D/E matrices
- [x] `experiments.py` — 8-level experiment ladder
- [ ] Run full ladder and analyze results

### Phase 2: Concordia Integration
- [ ] Wire AIF agents into Concordia's entity_agent system
- [ ] LLM + AIF hybrid: `format_aif_context_for_llm()` feeds AIF state
  into LLM prompts as "[Internal Assessment]" context
- [ ] Validate: standalone vs Concordia-integrated produce similar patterns

### Phase 3: LLM-as-Node
- [ ] Implement LLMPrior/LLMObservation nodes in Python
- [ ] Replace hardcoded A-matrix with LLMObservation
- [ ] Replace D-matrix priors with LLMPrior
- [ ] Compare learning curves: standalone AIF vs LLM+AIF hybrid

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

## JaxMARL Bridge: Fast GABM ↔ Slow LLM

### The Speed Gap

| System | Speed | Per experiment | Use for |
|--------|-------|---------------|---------|
| **JaxMARL on GPU** | ~10⁶ env steps/sec | Milliseconds | Parameter sweep, policy search |
| **Concordia + LLM** | ~200 API calls/sprint | 5-15 minutes | Validation, qualitative analysis |

This is a **10⁵-10⁶×** speed difference. The solution: multi-fidelity optimization.

### Architecture: GPU-Accelerated Agent-Based Model (GABM)

```
JaxMARL (GPU, fast)          Concordia (LLM, slow)
┌─────────────────┐         ┌─────────────────┐
│ SustainHub env   │         │ SustainHub sim  │
│ as MARL problem  │         │ with LLM agents │
│                  │         │                 │
│ State: (health,  │  best   │ Full natural    │
│  urgency, roles) │ params  │ language sprint │
│ Actions: tasks   │────────→│ planning and    │
│ Rewards: HI      │         │ decision-making │
│                  │         │                 │
│ AIF agents with  │         │ Level 7: LLM    │
│ A/B/C/D/E in JAX │         │ as factor graph │
│                  │  valid- │ node (RxInfer)  │
│ Levels 0-6 run   │←────── │                 │
│ here at 10⁶x     │  ation │ Qualitative     │
│ speed             │        │ insights        │
└─────────────────┘         └─────────────────┘
```

### Why JaxMARL?

- **Vectorized environments**: All agents step simultaneously in JAX
- **Batched episodes**: Run 1000s of SustainHub episodes in parallel on one GPU
- **Dictionary-based API**: `{agent_name: action}` matches our structure
- **Existing algorithms**: IPPO, MAPPO, QMIX ready to benchmark against AIF
- **JAX → active_inference.py port**: NumPy → JAX is near-trivial
  (`np.array` → `jnp.array`, add `@jax.jit`)

### The Bridge: `active_inference.py` in JAX

The A/B/C/D/E matrices are already pure NumPy. A JAX port enables:

```python
import jax.numpy as jnp
from jax import vmap, jit

# Vectorize over N_BATCH environments × N_AGENTS agents
batched_decide = vmap(vmap(select_action, in_axes=(None,None,None,None,None,0,None,None)),
                      in_axes=(None,None,None,None,None,0,None,None))

# Run 1000 environments × 4 agents in one GPU call
actions, probs = batched_decide(A, B, C, D, E, beliefs_batch, gamma, alpha)
```

This lets us:
1. Run experiment ladder levels 0-6 at GPU speed (seconds, not minutes)
2. Sweep AIF hyperparameters (gamma, alpha, learning_rate) over 10,000 configs
3. Find optimal A/B/C/D/E matrices that maximize SustainScore
4. Feed best configs into Level 7 (LLM validation)

### Connecting JaxMARL to Active Inference

JaxMARL environments use this API:
```python
obs, state = env.reset(key)
actions = {agent: policy(obs[agent]) for agent in env.agents}
obs, state, reward, done, infos = env.step(key, state, actions)
```

Our active inference agents become the `policy`:
```python
def aif_policy(obs, beliefs, A, B, C, D, E, gamma, alpha):
    # 1. Update beliefs from observation
    beliefs = update_beliefs(A, beliefs, obs)
    # 2. Select action via EFE
    action, probs = select_action(A, B, C, D, E, beliefs, gamma, alpha)
    return action, beliefs
```

### What JaxMARL Benchmarks Give Us

| JaxMARL Algorithm | What it tells us about SustainHub |
|------------------|----------------------------------|
| **IPPO** (Independent PPO) | Baseline: what if agents don't coordinate? |
| **MAPPO** (Multi-Agent PPO) | Upper bound: centralized training, decentralized execution |
| **QMIX** | How much does factored Q-values help vs full joint? |
| **AIF (ours)** | How does active inference compare to MARL algorithms? |

### Implementation Path

1. **Port SustainHub as JaxMARL environment** (`sustain_hub_env.py`)
   - State: project_health × task_urgency
   - Observations: per-agent partial observations
   - Actions: task selection
   - Rewards: SustainHub payoff structure
2. **Port `active_inference.py` to JAX** (`active_inference_jax.py`)
   - `@jax.jit` all core functions
   - `vmap` over agents and environments
3. **Run benchmarks**: IPPO vs MAPPO vs QMIX vs AIF levels 0-6
4. **Validate winners**: Feed best configs into Concordia + LLM (Level 7)

### Phase Transition Detection: When to Switch from System 1 → System 2

The JAX inner loop should run autonomously until it detects a **phase
transition** — a structural break indicating the RL agents' learned model
of the world is failing. Only then does it pause and invoke Concordia.

| Trigger | What it detects | Implementation |
|---------|----------------|----------------|
| **Policy Entropy Spike** | Agents lost confidence (H(π) → high) | `H = -Σ π(a|s) log π(a|s)` — rolling average across agents |
| **TD-Error Volatility** | "Surprise" — outcomes deviate from expectations | `Var(δ_t)` over rolling window; spike = model mismatch |
| **Critical Slowing Down** | System approaching collapse | Lag-1 autocorrelation or rolling variance of HI expanding |
| **CUSUM Changepoint** | Structural break in a metric | Cumulative sum of deviations from expected mean breaches threshold |
| **WeightWatcher α shift** | Network weight structure reorganized | HT-RMT spectral analysis on DRL weight matrices (requires deep nets) |

In active inference terms, these triggers are all forms of **precision
collapse** — the agent's confidence in its generative model drops below
a threshold, signaling that the "fast" System 1 model is no longer adequate
and the "slow" System 2 (LLM reasoning) must intervene.

```python
# JAX pseudocode for policy entropy trigger
@jax.jit
def should_wake_concordia(policy_probs, threshold=1.5):
    entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-8), axis=-1)
    mean_entropy = jnp.mean(entropy)
    return mean_entropy > threshold
```

### Prior Art: Dual-Process Social Simulations

| Framework | Fast Layer | Slow Layer | Bridge Mechanism |
|-----------|-----------|-----------|-----------------|
| **AgentTorch** | Tensor-based ABM (millions of agents) | LLM plug-in for adaptive behavior | Scale-aware activation |
| **Hawkes-Guided LLM** | Hawkes process (statistical timing) | LLM for contextual action content | Point-process triggers |
| **SimFleet Cognitive** | Standard ABM (physics/movement) | LLM for end-of-day reflection | Episodic memory bridge |
| **MARS** | RL policy (System 1) | Tool-using LLM (System 2) | GRPO concurrent optimization |
| **SustainHub (ours)** | JaxMARL (POMDP + AIF in JAX) | Concordia (LLM + memory + governance) | State translator + phase triggers |

### State Translation Layer

The bridge between System 1 (JAX) and System 2 (Concordia):

**Bottom-Up (JAX → Concordia)**:
JAX tensor `{agent_rewards, task_completion, HI, burnout}` →
Natural language Sprint Report injected into Concordia context window →
Triggers Community Retrospective scene

**Top-Down (Concordia → JAX)**:
Concordia agents debate and vote on policy →
Structured JSON output parsed →
Updated reward weights in JAX environment
(e.g., "Maintenance Premium" → `REWARD_NONPREFERRED_SUCCESS = 2.5`)

### Reference

- **JaxMARL**: https://github.com/FLAIROx/JaxMARL
  - 11 environments, 8 algorithms, all GPU-accelerated
  - Dictionary-based API matches SustainHub structure
  - SMAX (StarCraft II alternative) shows the vectorization approach
- **AgentTorch**: https://github.com/AgentTorch/AgentTorch
  - Tensor-based ABM scaling to millions of agents
- **GABM concept**: GPU-Accelerated Agent-Based Models for social simulation
  at population scale

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
