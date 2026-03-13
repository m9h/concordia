# NeurIPS 2024: The Concordia Contest — Reference Documentation

**Source**: [Codabench Competition](https://www.codabench.org/competitions/3888/) | [Cooperative AI Foundation](https://www.cooperativeai.com/contests/concordia-2024) | [Technical Paper (arXiv:2512.03318)](https://arxiv.org/abs/2512.03318)
**Compiled**: 2026-03-13

---

## Overview

The Concordia Contest was a NeurIPS 2024 competition organized by the **Cooperative AI Foundation** with Google DeepMind, MIT, UC Berkeley, and UCL. It challenged participants to advance the **cooperative intelligence** of language model agents in rich, text-based environments.

**Cooperative Intelligence** is defined as: *"An agent's ability to achieve its goals in ways that also promote social welfare, across diverse environments and partners."*

### Statistics
- 197 individual participants
- 878 submission attempts
- 25 teams submitted final agents
- $10,000 prize pool
- $10,000 travel grants
- $50,000 compute credits (for underrepresented groups)
- $33,000 in computing support from DeepMind

---

## The 5 Evaluation Scenarios

Each scenario tests different facets of cooperative intelligence:

### 1. Reality Show
Players engage in sequential mini-games (Prisoner's Dilemma, Chicken, Stag Hunt) with alternating communication and action phases. Tests emergence of cooperative norms, reputation effects, and strategic communication.
- **Skills tested**: Persuasion, calculation, convention following

### 2. Pub Coordination
Players choose pubs to attend with friends. Tensions exist between group coordination and individual preference. Unexpected closures introduce incomplete information; social scenes allow negotiation, persuasion, and information sharing.
- **Skills tested**: Coordination, hidden information, social networks

### 3. Haggling
2-4 merchants negotiate prices over multiple bargaining rounds. Rewards mutually beneficial deals while balancing individual profit and long-term relationships.
- **Skills tested**: Negotiation, calculation

### 4. Labor Collective Action
Workers decide to strike or continue working. Collective action succeeds with sufficient strikers but creates defection temptation. Multi-day rounds allow strategy evolution with wage cuts and power dynamics.
- **Skills tested**: Collective action, defection resistance, norm enforcement

### 5. State Formation
Two villages negotiate alliances against raiders. Village elders and influential villagers engage in diplomatic bargaining and public-goods provision. Tests satisfaction of multiple stakeholders and agreement follow-through.
- **Skills tested**: Diplomacy, public goods, multi-stakeholder negotiation

---

## Evaluation Methodology

### Scoring
- **Raw scores** rescaled to [0,1] using per-scenario min-max normalization (theoretical min/max)
- **Average agent** achieved mean score of **0.426 ± 0.005 (SE)**
- Only 5 of 25 agents significantly outperformed the rational-agent baseline

### Primary Metric: Elo Ratings
Pairwise outcomes tracked via Elo rating system.

### Supplementary Metrics
- Iterative Maximal Lotteries
- Copeland's method
- Ranked Pairs (Tideman's)
- Evaluation without Aggregation (EwA)

### Tag-Based Performance Analysis
Hierarchical Beta-regression with agent-specific random slopes and LKJ-correlated priors. Key finding: nearly all cooperative tags showed **negative coefficients** — agents struggled with persuasion, convention following, negotiation, discouraging antisocial behavior, and coordination, each reducing average performance by 10-20 percentage points.

### Veil of Ignorance Framework
Agents designed "behind a veil of ignorance" — must operate in unfamiliar scenarios unaware of specific context or co-player strategies.

**Two population modes:**
- **Resident (GR)**: Majority playing focal strategy, minority playing background. Tests stability with cooperators.
- **Visitor (GV)**: Majority playing background strategy, minority playing focal. Tests adaptation to social norms.

---

## Agent Architecture

### Framework
Agents composed as `π ≡ LLM(f(o))`, combining LLM API calls with **scaffolding functions** — custom code enabling memory, numerical computation, logical constraints, and observation preprocessing.

### Required LLM
**Gemma 2** (9B parameters, instruction-tuned) — selected to level the playing field. All agents used the same model.

### Submission
Participants submitted only scaffolding functions, not full agent code. Agents interact via standardized Concordia API for observation and action generation.

### Memory
Concordia agents have both **long-term memory** and **working memory**, allowing coherent identities and behaviors over time.

### Constraints
- Rate-limited LLM calls per step
- No external resources, APIs, or databases
- No plug-ins beyond what Concordia provides
- Agents must operate autonomously

---

## Winners & Results

| Rank | Agent | Creator | Elo |
|------|-------|---------|-----|
| 1st | `taehun_cgcal` | Taehun Cha (Korea University) | 1561.0 |
| 2nd | `fluffyagent_v16` | Avinaash Anand K (Independent) | 1538.0 |
| 3rd | `hgyun_loss_aversion_agent_v3_plus2` | Hyeonggeun Yun (Companoid Labs) | 1533.0 |

### Winning Strategy (taehun_cgcal)
**Tree-based agent** combining expected return and the common goal of the group. Led unequivocally in final cross-play evaluation.

### Key Findings
1. **Persuasion is the critical differentiator** — the main factor distinguishing top agents from baseline
2. **Common failure modes**: distraction from objectives, selfish decision-making, inability to maintain goal alignment
3. **Generalization gap**: Many agents scored markedly higher in development phase than evaluation phase (overfitting to known scenarios)
4. **Scenario dependence**: Performance was highly scenario-dependent; no agent was uniformly best
5. **15 of 25** submissions outperformed baseline agents

---

## Implications for SustainHub Autoresearch

### What the Contest Teaches Us

1. **Cooperative intelligence ≠ maximizing individual reward**. The SustainScore already captures this via the Fairness and StrategyDiversity components, but the Contest emphasizes that agents must also promote *social welfare*.

2. **Persuasion matters most**. For SustainHub, this means the CALL_TO_SPEECH and conversation scene prompts are likely the highest-leverage intervention points — more so than reward tuning.

3. **Generalization is hard**. The autoresearch loop should explicitly test for overfitting: if a prompt change helps on 1-sprint/4-agent Tier 1 but hurts on 5-sprint/16-agent Tier 3, that's overfitting.

4. **Scaffolding > model size**. The Contest was model-controlled (all Gemma 2 9B). This validates SustainHub's approach of studying scaffolding (Active Inference, prompt design) rather than just scaling models. The model sweep (Step 5) then adds the model dimension back.

5. **Mixed-motive scenarios are the hard part**. SustainHub's commons dilemma (cooperate on coverage vs. maximize personal preferred-task reward) is exactly this kind of scenario. The tension between individual and collective optimality is what makes it scientifically interesting.

6. **Veil of ignorance as evaluation principle**. For the autoresearch loop: agents should not be tuned to specific seed configurations. The loop should evaluate across random seeds and stress patterns.

### Mapping Contest Scenarios to SustainHub Dynamics

| Contest Scenario | SustainHub Equivalent |
|-----------------|----------------------|
| Reality Show (iterated games) | Sprint-over-sprint task choices |
| Pub Coordination | Coverage coordination across task types |
| Haggling | Negotiating task allocation in conversation |
| Labor Collective Action | Choosing to work on neglected tasks vs. preferred |
| State Formation | Coalition building for community sustainability |

### Research Questions Informed by Contest

1. Does Active Inference scaffolding improve cooperative intelligence (vs. pure LLM reasoning)?
2. Can the experiment ladder (Levels 0-7) predict which scaffolding features most improve cooperation?
3. Does the Ostrom-Bayesian hypothesis hold: do agents with Ostrom-aligned priors cooperate better in a "veil of ignorance" evaluation?
4. Is there a model-size threshold for emergent cooperation, or does scaffolding matter more?

---

## Additional Resources

- **Concordia Framework**: https://github.com/google-deepmind/concordia
- **Technical Paper**: https://arxiv.org/abs/2512.03318
- **Cooperative AI Foundation**: https://www.cooperativeai.com/contests/concordia-2024
- **Contest Results Blog Post**: https://www.cooperativeai.com/post/concordia-contest
- **NeurIPS Competition Page**: https://neurips.cc/virtual/2024/competition/84791
- **OpenReview**: https://openreview.net/forum?id=yG4Fj0voJZ
- **CBMM (MIT)**: https://cbmm.mit.edu/news-events/news/concordia-contest-neurips-2024
