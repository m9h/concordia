# SustainHub Autoresearch

Autonomous optimization of an LLM-powered multi-agent social simulation,
inspired by [autoresearch](https://github.com/karpathy/autoresearch).

The goal is not just to maximize a number — it's to discover simulation designs
that produce **realistic, interesting emergent social dynamics** where LLM agents
genuinely grapple with commons dilemmas.

## Setup

1. **Run tag**: propose a tag (e.g. `mar11`). Branch: `autoresearch/<tag>`.
2. **Create branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read in-scope files**: `simulation.py`, `scenario_config.py`, `social_data.py`, `tools.py`, `run.py`.
4. **Verify environment**: `.venv` exists, `GEMINI_API_KEY` set.
5. **Initialize `results.tsv`** with header row.
6. **Run baseline**, then begin the loop.

## The Composite Metric

Raw Harmony Index is gameable (make everyone cooperate trivially → HI=1.0 but
boring). Instead, optimize **SustainScore**, a composite that rewards
scientifically interesting dynamics:

```
SustainScore = HI × (1 + RQ) × Fairness × StrategyDiversity × StressValidity
```

Where:
- **HI** (Harmony Index, 0-1): community sustainability
- **RQ** (Resilience Quotient, 0-1): recovery from stress events
- **Fairness** = `1 - normalized_score_variance` (0-1): how equitably rewards are distributed. Gini coefficient of agent scores, inverted.
- **StrategyDiversity** = fraction of agents that changed task type across sprints (0-1). Higher = agents are actually adapting, not stuck.
- **StressValidity** = `|HI_stressed - HI_unstressed| > 0.05 ? 1.0 : 0.5`. Stress scenarios should actually matter. If turning stress on/off doesn't change outcomes, the simulation isn't modeling real dynamics.

**Higher SustainScore = better.** Target: >0.60 is good, >0.80 is excellent.

The `evaluate.py` script computes this automatically from `results.json`.

## What You CAN Modify

| File | What's in it | Experiment ideas |
|------|-------------|-----------------|
| `simulation.py` | Sprint logic, payoff engine, conversation scenes, scoring | Change how HI is calculated; add retrospective phase; modify information flow between agents |
| `scenario_config.py` | Sprints, agent selection, stress toggle | Scale up N, change conditions |
| `social_data.py` | Profiles, rewards, tasks, relationships, prompts | Reward shaping; personality tuning; prompt engineering for richer dialogue |
| `tools.py` | Agent tools | New tools; tool effectiveness tuning |

### Key Variation Axes (ordered by expected impact)

**Layer 1 — Prompt Engineering** (highest signal, cheapest to try)
- `CALL_TO_SPEECH`, `DECISION_PREMISE`, `SPRINT_PREMISES` in `social_data.py`
- The framing prompts are the #1 lever for agent behavior quality
- Try: explicit mention of trade-offs, reference to past outcomes, social pressure cues

**Layer 2 — Reward Shaping** (`social_data.py`)
- `REWARD_PREFERRED_SUCCESS=3.0` vs `REWARD_NONPREFERRED_SUCCESS=1.0` — is 3:1 the right tension?
- Add: mentoring bonus, collective coverage bonus, neglected-area penalty
- Try: diminishing returns on repeated preferred tasks

**Layer 3 — Simulation Mechanics** (`simulation.py`)
- Payoff engine: HI formula, decay rates, coverage bonuses
- Sprint structure: add retrospective/reflection phase
- Information design: what agents know about others' choices (full, partial, none)
- Memory: how agents use past sprint outcomes to inform decisions

**Layer 4 — Agent Design** (`social_data.py`)
- Backstory wording shapes LLM role-play (small changes → big behavioral shifts)
- Team composition: role ratios, expertise mix
- Personality trait balance: cooperative vs competitive teams

**Layer 5 — Stress & Resilience** (`social_data.py`)
- Scenario severity, timing, warning mechanisms
- Recovery mechanics after disruption

## What You CANNOT Modify

- `run.py` (evaluation harness)
- Concordia framework files (outside `examples/games/sustain_hub/`)
- `pyproject.toml` dependencies

## Experiment Tiers

Run experiments at the appropriate scale. Start small, validate, then scale up.

### Tier 1: Fast Iteration (development)
```bash
.venv/bin/python3 -m examples.games.sustain_hub.run \
  --model_name='gemini-2.0-flash' \
  --num_sprints=1 --community_size=4 --skip_backstory \
  --output_dir=/tmp/sustain_hub_autoresearch > run.log 2>&1
```
- **4 agents, 1 sprint, ~2 min, ~$0.03/run**
- Use for: reward tuning, prompt experiments, crash testing
- ~200 LLM calls per run

### Tier 2: Validation (confirm improvements)
```bash
.venv/bin/python3 -m examples.games.sustain_hub.evaluate --runs=3 --log \
  --description="your experiment description"
```
- **3 averaged runs to handle stochasticity**
- Use for: confirming Tier 1 winners before keeping

### Tier 3: Full Scale (overnight / DGX Spark)
```bash
.venv/bin/python3 -m examples.games.sustain_hub.run \
  --model_name='gemini-2.0-flash' \
  --num_sprints=5 --community_size=16 --enable_stress \
  --output_dir=/tmp/sustain_hub_full > run.log 2>&1
```
- **16 agents, 5 sprints, stress enabled, ~15-25 min, ~$0.12/run**
- Use for: final validation, paper-worthy results
- ~2000+ LLM calls per run

## Scaling the Simulation

### Model Backend Options

The simulation is **LLM-call-bound**, not GPU-bound. The bottleneck is
latency × number of sequential LLM calls (~200-400 per sprint).

| Backend | Speed | Cost | Best for |
|---------|-------|------|----------|
| **Gemini 2.0 Flash Lite** (API) | ~350 tok/s, $0.075/M input | ~$0.03/run | Fast iteration, cheapest |
| **Gemini 2.0 Flash** (API) | ~300 tok/s, $0.10/M input | ~$0.05/run | Default, good quality |
| **Qwen 2.5-7B** (local, vLLM/SGLang) | 100-370 tok/s | $0 marginal | Overnight loops, no rate limits |
| **Qwen 3.5-4B** (local) | 200-500 tok/s | $0 marginal | Maximum speed, minimum viable quality |
| **Ollama** (local, any model) | ~40 tok/s | $0 marginal | Prototyping only (poor concurrency) |

Concordia has built-in backends for all of these:
`concordia/contrib/language_models/{ollama,vllm,groq,together}/`

To switch: change `run.py` to instantiate a different model class, or add a
`--backend` flag.

### Making Simulations Faster

**Estimated speedups** (cumulative):

| Optimization | Speedup | Effort | How |
|-------------|---------|--------|-----|
| Switch Sequential → **Simultaneous engine** | 4-8x | Medium | Agents act in parallel per round |
| Use **Gemini Flash Lite** instead of Flash | 1.3x | Trivial | Change `--model_name` |
| **Skip backstory** generation | 1.2x | Trivial | `--skip_backstory` flag |
| **Component caching** (skip pre_act if no new observations) | 2-3x | Medium | Modify entity_agent.py |
| **Batch GM calls** (combine next_acting + action_spec) | 1.5x | Medium | Modify sequential.py |
| **Local model** on DGX Spark (no network latency) | 2-5x | Medium | vLLM/SGLang serving |
| **Theoretical max** | **25-100x** | | Enables 16 agents × 10 sprints in ~2 min |

### Scaling to Larger N

Current bottleneck: conversation scenes have O(N) sequential rounds
(each agent speaks in turn). With the **Asynchronous engine**
(`concordia/environment/engines/asynchronous.py`), agents run in independent
threads — O(1) wall-clock time regardless of N.

Target: **64-agent simulations completing in <10 minutes** on DGX Spark
with local Qwen 2.5-7B via SGLang.

### DGX Spark Configuration

The NVIDIA DGX Spark (Grace Blackwell GB10, 128GB unified memory) can:
- Run Qwen 2.5-7B at ~370 tok/s decode (batch 32)
- Run Qwen 3.5-4B at ~500+ tok/s
- Handle models up to 200B parameters
- Process all 16 agents' decisions in parallel via batched inference

Setup:
```bash
# On DGX Spark:
pip install sglang  # or vllm
python -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --port 8000

# Point Concordia at it via the vLLM or OpenAI-compatible backend
```

## Logging Results

Tab-separated `results.tsv`:

```
commit	sustain_score	harmony_index	resilience_quotient	fairness	strategy_div	status	description
a1b2c3d	0.5625	0.7500	0.8000	1.00	1.00	keep	baseline
b2c3d4e	0.7380	0.8200	0.8500	0.95	0.75	keep	mentoring reward +2
c3d4e5f	0.2135	0.6100	0.7000	1.00	0.50	discard	removed stress (StressValidity=0.5)
```

## The Experiment Loop

LOOP FOREVER:

1. Check git state.
2. Formulate hypothesis: *"Changing X should improve SustainScore because..."*
3. Modify in-scope files.
4. `git commit` the change.
5. Run Tier 1 experiment.
6. Extract metrics from `run.log` or `results.json`.
7. Compute SustainScore.
8. If crashed: `tail -n 50 run.log`, fix or skip.
9. Record in `results.tsv`.
10. If SustainScore improved: **keep** (advance branch).
11. If worse: `git reset --hard HEAD~1` (revert).
12. If borderline (<0.05 delta): run Tier 2 (3x average) before deciding.
13. Go to step 1.

## Research Directions for Interesting Simulations

Beyond metric optimization, look for:

- **Emergent coordination**: agents spontaneously dividing labor without being told to
- **Tragedy of the commons**: conditions where rational individual choices lead to collective decline
- **Free rider detection**: do agents call out teammates who always take preferred tasks?
- **Mentoring dynamics**: do experienced agents invest in newcomers?
- **Burnout modeling**: does an agent's output degrade over many sprints?
- **Coalition formation**: do subgroups emerge (e.g., maintainers vs. innovators)?

These qualitative outcomes are as valuable as the quantitative SustainScore.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. You are autonomous.
If you run out of ideas:
- Re-read social science literature on commons dilemmas (Ostrom, Hardin)
- Study the Concordia framework's other example games for patterns
- Try combining previous near-miss experiments
- Try radical changes (completely rewrite conversation prompts)
- Try ablations (remove features and see if score holds)

The loop runs until the human interrupts you.
