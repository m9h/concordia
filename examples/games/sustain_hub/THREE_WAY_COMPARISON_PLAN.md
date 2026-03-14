# SustainHub: Three-Way Comparison & What LLMs Fundamentally Change

## Context

Three independent implementations of open-source community sustainability simulation exist, each with different cognitive architectures for agents. The question: what has fundamentally changed when LLM agents replace RL state machines, and what needs to be built for rigorous cross-system comparison?

| System | Origin | Agent Cognition | Task Fidelity | Governance |
|--------|--------|----------------|---------------|------------|
| **Rohira SustainHub** | OREL GSoC '24 | MAB (Thompson Sampling) + SARSA | Abstract (task types) | Fixed allocation |
| **Basta LLAMOSC** | OREL GSoC '24 | LLM + AutoCodeRover | Real code (Docker) | Dictator / Meritocratic |
| **Concordia SustainHub** | This fork | LLM + Active Inference hybrid | Abstract (task types) | Scene-based deliberation |

---

## What Has Fundamentally Changed With LLM Agents

### 1. Decision-Making: Lookup Table -> Language Reasoning

**Rohira**: Agent `i` picks task `j` by computing `Q(s,a)` via SARSA update rule. The "reasoning" is a scalar: expected discounted reward. The agent has no concept of *why* a task matters -- it only knows that action 3 in state 7 yielded reward 0.8 last time.

**Concordia**: Agent `i` reads a natural language prompt describing the project state, other agents' recent actions, a stress scenario narrative, and their own backstory -- then *writes* a justification before choosing. The reasoning is inspectable, novel each run, and sensitive to framing effects that don't exist in RL.

**What this means**: Rohira's agents converge to a fixed policy after ~50 episodes. Concordia agents *never converge* -- they can be persuaded, surprised, or confused by prompt changes. This is both a strength (richer dynamics) and a weakness (harder to reproduce). The autoresearch loop exploits this: each prompt variation is a new "policy" without retraining.

### 2. State Representation: Feature Vector -> Natural Language Context

**Rohira**: State = `(agent_expertise, task_difficulty, current_HI, sprint_number)` -- a 4-tuple. Context is a fixed-dimension vector.

**Concordia**: State = a multi-paragraph narrative constructed by the Game Master, including: project health metrics, stress scenario description, other agents' stated intentions, the agent's own memory of past sprints, and an optional Active Inference "[Internal Assessment]" block with Bayesian posteriors. The state space is effectively infinite.

**What this means**: Rohira's proposed Contextual Thompson Sampling (GSoC 2025) adds context features to bandit arms -- but the "context" is still a fixed feature vector. In Concordia, the context is *everything the LLM can read*, including social dynamics that can't be reduced to features. The LLM can notice patterns ("everyone is free-riding") that no feature engineer anticipated.

### 3. Communication: None -> Rich Social Interaction

**Rohira**: Agents are isolated decision-makers. No communication, no persuasion, no coordination beyond observing shared HI.

**LLAMOSC**: Agents discuss in a shared channel. An LLM-powered "maintainer" agent assigns tasks or agents self-select. Communication is the governance mechanism.

**Concordia**: Agents participate in structured scenes -- a "Sprint Planning" discussion where they state intentions, hear others' plans, and potentially adjust. The Game Master mediates. Communication produces emergent phenomena: coalition formation, social pressure, free-rider shaming -- none of which exist in Rohira's RL agents.

**What this means**: Rohira's metrics (HI, RQ) measure *outcomes* but can't capture *process*. When Concordia agents achieve HI=0.85, the transcript shows *how* -- did one agent convince others to cover neglected tasks? Did social pressure work? These process-level findings are the unique contribution of LLM simulation.

### 4. Task Execution: Abstract Reward -> (Potentially) Real Code

**Rohira**: `reward = f(task_type, agent_expertise, task_difficulty)` -- a deterministic function. A "senior developer" doing a "hard bugfix" always scores the same.

**LLAMOSC**: AutoCodeRover generates real code patches in Docker. The reward is grounded in actual code quality (tests pass/fail, patch correctness). Task difficulty is *real* -- some bugs are genuinely harder.

**Concordia (current)**: Same abstract reward as Rohira. The `AutoCodeRover` in `tools.py` is a stub returning "85% confidence" -- it doesn't execute real code. This is the biggest gap vs. LLAMOSC.

**What this means**: Our simulation currently has the richest agent *cognition* (LLM + Active Inference) but the most abstract task *execution*. For the coding-tasks comparison to be meaningful, we either need real code generation (SWE-agent/Aider integration) or we need to be explicit that we're studying *social dynamics* rather than *code quality*.

### 5. Learning: Stateful Convergence -> Stateless In-Context

**Rohira**: SARSA updates Q-tables across episodes. Agents genuinely *learn* -- episode 100 is different from episode 1. Proposed SARSA(lambda) with eligibility traces accelerates this.

**Concordia**: LLM agents don't learn across runs. Each simulation starts fresh. "Learning" happens only within a single run via the agent's memory component (they remember what happened in sprint 1 when deciding in sprint 3). The Active Inference layer updates beliefs *within* a run but resets between runs.

**What this means**: Rohira can study convergence dynamics and learning curves. We study *single-run emergent behavior*. These are complementary -- Rohira answers "what does the optimal policy look like?", we answer "what happens when agents reason about the commons in natural language?". The 8-level experiment ladder bridges them: Levels 0-6 do converge (pure AIF), Level 7 (LLM) does not.

### 6. Metrics: Matched Core, Divergent Extensions

| Metric | Rohira | Concordia | Compatible? |
|--------|--------|-----------|-------------|
| Harmony Index (HI) | `mean(individual_HI)` | Same formula | Yes -- directly comparable |
| Resilience Quotient (RQ) | HI recovery after stress | Same concept | Yes, if stress events align |
| SustainScore | N/A | `HI * (1+RQ) * Fairness * StratDiv * StressValidity` | Concordia-only composite |
| Burnout Risk Score (BRS) | Proposed GSoC '25 | Not yet implemented | **Step 2** |
| Skill Utilization Efficiency (SUE) | Proposed GSoC '25 | Not yet implemented | **Step 2** |
| Community Health Score (CHS) | Proposed GSoC '25 | Not yet implemented | **Step 2** |
| Gini coefficient | N/A | Fairness component | Rohira could add |
| Strategy Diversity | N/A | Simpson's index | Concordia-only |
| Transcript analysis | N/A | Full conversation logs | Concordia-only |

---

## Implementation Plan

### Step 1: Establish Matched Evaluation Protocol

**Goal**: Make HI and RQ directly comparable across all three systems.

**Files to modify**: `evaluate.py`, `social_data.py`

1. **Align reward function**: Verify our `action_to_scores()` matches Rohira's reward formula. Key differences to resolve:
   - Rohira uses `preferred_task_bonus * expertise_multiplier` -- confirm our `REWARDS` dict maps equivalently
   - Rohira's expertise levels (Novice/Intermediate/Expert) -> our (Apprentice/Intermediate/Senior/Expert) -- document mapping
   - Rohira's task difficulty is drawn from a distribution; ours is implicit in task type -- decide whether to add explicit difficulty

2. **Align HI computation**: Both use `mean(individual_HI)` where individual HI considers task coverage, workload balance, and skill match. Verify the weighting is identical or document differences.

3. **Align stress events**: Rohira uses "contributor dropout" (remove 20% of agents). We have 6 stress types. For comparison, run with only `contributor_dropout` stress at matching timing.

4. **Matched community sizes**: Rohira tested 10, 20, 30 agents. We've tested 4, 8. Add runs at 10 and 20.

5. **Matched sprint counts**: Rohira runs 50+ episodes for convergence. Our LLM runs are expensive (~72s per sprint). For comparison, report HI at matched sprint counts (sprint 5, 10) and note that RL agents continue to improve while LLM agents don't.

### Step 2: Add Rohira's Proposed Metrics (GSoC 2025)

**Goal**: Implement BRS, SUE, CHS so we can report them alongside Rohira's future results.

**Files to modify**: `evaluate.py`

1. **Burnout Risk Score (BRS)**: Track per-agent consecutive non-preferred task assignments. `BRS_i = consecutive_nonpreferred_sprints / total_sprints`. Add to sprint_history recording.

2. **Skill Utilization Efficiency (SUE)**: `SUE = mean(skill_match_score)` where skill_match is 1.0 for preferred task, 0.5 for adjacent, 0.0 for unrelated. We already have expertise levels -- need to define the adjacency matrix.

3. **Community Health Score (CHS)**: Rohira proposes `CHS = w1*HI + w2*(1-mean_BRS) + w3*SUE + w4*RQ`. Implement as a composite metric in evaluate.py alongside SustainScore.

### Step 3: Add Governance Modes (LLAMOSC Comparison)

**Goal**: Enable direct comparison with LLAMOSC's governance experiment.

**Files to modify**: `simulation.py`, `social_data.py`, `run.py`

LLAMOSC tested three governance models:
1. **Dictator**: A maintainer assigns tasks to agents
2. **Meritocratic**: Tasks assigned based on expertise/track record
3. **Free choice**: Agents self-select (our current default)

Implementation:
1. Add a `--governance` flag to `run.py` with values `free_choice`, `dictator`, `meritocratic`
2. **Dictator mode**: Add a "Project Lead" GM scene before task selection that assigns tasks. The GM prompt says "assign tasks to maximize coverage based on agent expertise." Agent action space becomes `{accept, negotiate, refuse}` instead of task selection.
3. **Meritocratic mode**: Limit agent action space based on past performance. Agents who performed well on a task type get priority access. Implementation: sort agents by cumulative reward per task type, assign top-N agents to each task type, remaining agents choose freely.
4. Compare HI/RQ across governance modes -- this is LLAMOSC's core finding reproduced in a richer cognitive architecture.

### Step 4: Real Code Generation Integration (Closing the LLAMOSC Gap)

**Goal**: Replace the AutoCodeRover stub with actual code generation capability.

**Files to modify**: `tools.py`, `simulation.py`

Options (ordered by implementation effort):
1. **SWE-agent integration** (preferred): SWE-agent is open-source, Docker-based, and has a Python API. Replace the stub with a call to SWE-agent's `run()` on a curated set of GitHub issues from a test repository. Reward = `1.0` if tests pass, `0.0` otherwise.
2. **Aider integration**: Aider has a simpler API (`aider --message "fix bug X"`). Less autonomous than SWE-agent but faster.
3. **Grounded difficulty scoring**: Keep abstract tasks but ground difficulty in real data -- analyze a corpus of GitHub issues to calibrate how long bugfixes vs. features vs. docs take for junior/senior developers. Use these distributions as reward noise.

**Decision**: Option 3 is sufficient for the social dynamics research question. Options 1-2 are needed only if we want to reproduce LLAMOSC's code-quality findings. Recommend starting with Option 3, adding Option 1 as a stretch goal.

### Step 5: Run Comparison Experiments

**Prerequisites**: DGX Spark with vLLM serving a capable model (Qwen2.5-7B minimum)

**Experiment matrix**:

| Experiment | System | Agents | Sprints | Stress | Seeds | Purpose |
|-----------|--------|--------|---------|--------|-------|---------|
| A1 | Concordia (LLM only) | 10 | 10 | dropout only | 5 | Match Rohira's setup |
| A2 | Concordia (LLM+AIF) | 10 | 10 | dropout only | 5 | AIF contribution |
| A3 | Concordia (AIF only, Level 4) | 10 | 50 | dropout only | 10 | Convergence comparison |
| B1 | Concordia free-choice | 8 | 5 | all 6 types | 5 | Governance baseline |
| B2 | Concordia dictator | 8 | 5 | all 6 types | 5 | vs. LLAMOSC dictator |
| B3 | Concordia meritocratic | 8 | 5 | all 6 types | 5 | vs. LLAMOSC meritocratic |
| C1 | Concordia (LLM+AIF+Ostrom) | 10 | 10 | dropout only | 5 | Ostrom contribution |

**Key comparisons**:
- A1 vs. Rohira's reported HI=0.81: Does LLM reasoning help or hurt?
- A2 vs. A1: Does AIF guidance improve LLM agent decisions?
- A3 vs. Rohira's convergence curves: Same math, different framework -- results should match
- B1/B2/B3 vs. LLAMOSC governance results: Same governance variation, different cognitive architecture
- C1 vs. A2: Ostrom priors as Bayesian formalization of governance principles

### Step 6: Write Comparison Paper / Report

**Output**: A structured comparison document covering:
1. **Architectural taxonomy**: RL vs. LLM vs. Hybrid cognitive architectures for social simulation
2. **Matched metric comparison**: HI, RQ, BRS, SUE tables across systems
3. **Unique contributions per system**:
   - Rohira: Convergence analysis, learning curves, scalability (30+ agents cheaply)
   - LLAMOSC: Grounded code quality, governance model comparison
   - Concordia: Process-level analysis (transcripts), Active Inference bridge, Ostrom formalization, autoresearch
4. **Multi-fidelity argument**: These aren't competitors -- they're fidelity levels. Use Rohira for fast parameter sweeps (10^6 episodes), Concordia for process understanding (10^2 runs), LLAMOSC for ground truth (10 runs). Results should be consistent across fidelity levels.

---

## Critical Files

| File | Modification | Purpose |
|------|-------------|---------|
| `evaluate.py` | Add BRS, SUE, CHS metrics | Metric alignment with Rohira |
| `simulation.py` | Add governance modes | LLAMOSC comparison |
| `social_data.py` | Add governance prompts, align rewards | Cross-system compatibility |
| `tools.py` | Replace AutoCodeRover stub | Code generation grounding |
| `run.py` | Add --governance flag | Experiment control |
| `experiments.py` | Add cross-system comparison experiments | Automated comparison runs |
| `bin/autoresearch.py` | Add governance variation axis | Autoresearch over governance modes |

## Verification Checklist

- [ ] **Metric alignment**: Run AIF Level 4 (no LLM) for 50 sprints with 10 agents, dropout stress at sprint 10 -- HI should be in Rohira's range (0.75-0.85) if reward functions match
- [ ] **Governance modes**: Run all 3 governance modes with `--use_mock` -- verify agents receive different action spaces and GM assigns tasks in dictator mode
- [ ] **BRS/SUE/CHS**: Run a 5-sprint simulation -- verify new metrics appear in results.json with plausible values
- [ ] **Cross-system comparison**: Generate comparison table from experiments A1-A3, B1-B3, C1 -- verify all metrics computed and formatted
- [ ] **Full DGX validation**: Run experiment A1 (10 agents, 10 sprints, 5 seeds) on vLLM -- ~6 hours, produces publishable HI/RQ numbers
