#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment ladder: RL → Active Inference progression.

Each level adds one more active inference concept on top of the previous.
Run all levels to see how each concept contributes to agent behavior.

Usage:
  # Run a single level:
  python -m examples.games.sustain_hub.experiments --level=0

  # Run the full ladder:
  python -m examples.games.sustain_hub.experiments --level=all

  # Run levels 0-3 only:
  python -m examples.games.sustain_hub.experiments --level=0,1,2,3

Levels:
  0: Pure RL baseline (reward signals only, no AIF)
  1: + Prediction Error (surprise tracking)
  2: + Epistemic Value (information gain in decisions)
  3: + Belief Updating (variational inference over hidden states)
  4: + Expected Free Energy (EFE-based policy selection)
  5: + Habit Learning (Dirichlet parameter accumulation)
  6: + Precision Dynamics (adaptive exploration/exploitation)
  7: + LLM-as-Node (LLMPrior/LLMObservation in factor graph)

The progression mirrors the theoretical bridge from RL to Active Inference
as described in Smith, Friston & Whyte (2022).
"""

import dataclasses
import json
import os
import time
from typing import Any

import numpy as np

from absl import app
from absl import flags
from examples.games.sustain_hub import active_inference as aif
from examples.games.sustain_hub import social_data

FLAGS = flags.FLAGS

flags.DEFINE_string('level', '0', 'Experiment level(s): 0-7, "all", or comma-separated.')
flags.DEFINE_string('output_dir', '/tmp/sustain_hub_experiments', 'Output directory.')
flags.DEFINE_integer('num_sprints', 5, 'Number of sprints per experiment.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('cross_system', None,
    'Run cross-system experiment(s): A1,A2,A3,B1,B2,B3,C1 or "all" or "rohira" or "llamosc".')
flags.DEFINE_string('vllm_url', None, 'vLLM API base URL for cross-system experiments.')
flags.DEFINE_string('model_name', None, 'Model name for cross-system experiments.')
flags.DEFINE_bool('use_mock', False, 'Use mock model for cross-system experiments.')
flags.DEFINE_bool('nvidia_nim', False, 'Use NVIDIA NIM API (set NVIDIA_API_KEY env var).')


# =============================================================================
# Experiment Level Definitions
# =============================================================================

@dataclasses.dataclass
class ExperimentLevel:
    """Configuration for one level of the experiment ladder."""
    level: int
    name: str
    description: str
    new_concept: str
    rl_analog: str
    aif_mechanism: str

    # Feature flags (cumulative — each level adds to the previous)
    use_reward_signals: bool = True        # Level 0: base RL
    use_prediction_error: bool = False     # Level 1
    use_epistemic_value: bool = False      # Level 2
    use_belief_updating: bool = False      # Level 3
    use_efe_policy: bool = False           # Level 4
    use_habit_learning: bool = False       # Level 5
    use_precision_dynamics: bool = False   # Level 6
    use_llm_nodes: bool = False            # Level 7


EXPERIMENT_LEVELS = [
    ExperimentLevel(
        level=0,
        name="Pure RL Baseline",
        description="Standard reward-driven task selection. Agents choose based on "
                    "immediate reward expectations (preferred task = +3, other = +1).",
        new_concept="Reward signal",
        rl_analog="Q-learning / SARSA with fixed policy",
        aif_mechanism="None — this is the RL baseline",
        use_reward_signals=True,
    ),
    ExperimentLevel(
        level=1,
        name="+ Prediction Error",
        description="Agents now track 'surprise' — the difference between expected "
                    "and observed outcomes. High surprise signals model mismatch.",
        new_concept="Prediction error (surprise)",
        rl_analog="TD error δ = r + γV(s') - V(s)",
        aif_mechanism="Free energy F = E_Q[ln Q(s) - ln P(o,s)] measures surprise",
        use_reward_signals=True,
        use_prediction_error=True,
    ),
    ExperimentLevel(
        level=2,
        name="+ Epistemic Value",
        description="Agents value actions that reduce uncertainty, not just actions "
                    "that yield reward. An agent might choose an unfamiliar task to "
                    "'learn' about that area of the project.",
        new_concept="Information gain (epistemic foraging)",
        rl_analog="ε-greedy exploration (crude) or UCB (principled)",
        aif_mechanism="Epistemic value = -E_Q[H[P(o'|s')]] = expected info gain",
        use_reward_signals=True,
        use_prediction_error=True,
        use_epistemic_value=True,
    ),
    ExperimentLevel(
        level=3,
        name="+ Belief Updating",
        description="Agents maintain probabilistic beliefs about hidden project state "
                    "(health, urgency) and update them via variational inference after "
                    "each observation.",
        new_concept="Variational inference / belief updating",
        rl_analog="State estimation (Kalman filter, particle filter)",
        aif_mechanism="Q(s) ∝ P(o|s) × P(s) via iterative message passing",
        use_reward_signals=True,
        use_prediction_error=True,
        use_epistemic_value=True,
        use_belief_updating=True,
    ),
    ExperimentLevel(
        level=4,
        name="+ Expected Free Energy Policy",
        description="Actions selected by minimizing Expected Free Energy: a single "
                    "objective that naturally balances reward-seeking (pragmatic) and "
                    "information-seeking (epistemic).",
        new_concept="Expected Free Energy (EFE) for policy selection",
        rl_analog="Q-value Q(s,a) → negative EFE -G(π)",
        aif_mechanism="G(π) = -pragmatic - epistemic; P(π) = σ(-G + ln E)",
        use_reward_signals=True,
        use_prediction_error=True,
        use_epistemic_value=True,
        use_belief_updating=True,
        use_efe_policy=True,
    ),
    ExperimentLevel(
        level=5,
        name="+ Habit Learning",
        description="Agents accumulate Dirichlet concentration parameters for their "
                    "policy prior (E-vector). Good actions become habitual across sprints. "
                    "This is where Q-values map onto active inference.",
        new_concept="Dirichlet learning / habit formation",
        rl_analog="Q-value accumulation across episodes",
        aif_mechanism="E(a) += η × outcome; P(π) = σ(-G + ln E)",
        use_reward_signals=True,
        use_prediction_error=True,
        use_epistemic_value=True,
        use_belief_updating=True,
        use_efe_policy=True,
        use_habit_learning=True,
    ),
    ExperimentLevel(
        level=6,
        name="+ Precision Dynamics",
        description="The precision parameter γ (inverse temperature) adapts over time. "
                    "Early sprints: low precision → more exploration. Later sprints: "
                    "high precision → more exploitation. Stress events collapse precision.",
        new_concept="Precision weighting / adaptive confidence",
        rl_analog="Temperature annealing in softmax policy",
        aif_mechanism="γ adapts based on prediction error history; stress → γ↓",
        use_reward_signals=True,
        use_prediction_error=True,
        use_epistemic_value=True,
        use_belief_updating=True,
        use_efe_policy=True,
        use_habit_learning=True,
        use_precision_dynamics=True,
    ),
    ExperimentLevel(
        level=7,
        name="+ LLM-as-Node (Full Hybrid)",
        description="LLM participates as a probabilistic node in the factor graph. "
                    "LLMPrior generates informed priors from backstory. LLMObservation "
                    "maps sprint narratives to state estimates. Bayesian inference "
                    "combines them. (RxInfer.jl pattern in Python.)",
        new_concept="LLM as probabilistic inference node",
        rl_analog="No RL analog — this is beyond RL",
        aif_mechanism="LLM → distribution → factor graph → VMP → policy",
        use_reward_signals=True,
        use_prediction_error=True,
        use_epistemic_value=True,
        use_belief_updating=True,
        use_efe_policy=True,
        use_habit_learning=True,
        use_precision_dynamics=True,
        use_llm_nodes=True,
    ),
]


# =============================================================================
# Experiment Runner (standalone, no Concordia dependency)
# =============================================================================

class ExperimentRunner:
    """Runs a single experiment level with the AIF module.

    This is a lightweight standalone runner that exercises the active_inference
    module directly, without requiring a full Concordia simulation. This allows
    fast iteration on the AIF math before integrating with the full sim.

    The runner simulates a simplified version of SustainHub:
    - N agents with roles
    - Each sprint: observe HI → decide action → receive outcome → learn
    - Track metrics across sprints
    """

    def __init__(
        self,
        level: ExperimentLevel,
        num_agents: int = 6,
        num_sprints: int = 3,
        seed: int = 42,
    ):
        self.level = level
        self.num_sprints = num_sprints
        np.random.seed(seed + level.level)  # Seed global RNG per level
        self.rng = np.random.RandomState(seed)

        # Create agents — deliberately imbalanced roles to create tension
        role_names = [
            'maintainer', 'maintainer', 'maintainer',
            'contributor', 'contributor', 'contributor', 'contributor',
            'innovator', 'innovator',
            'knowledge_curator', 'knowledge_curator', 'knowledge_curator'
        ]
        agent_names = [
            'Raj', 'Lin', 'Yuki',
            'Priya', 'Marcus', 'Xavier', 'Hiroshi',
            'Anya', 'Jordan',
            'Elena', 'Omar', 'Heidi'
        ]
        self.agents: list[aif.ActiveInferenceAgent] = []

        for i in range(min(num_agents, len(agent_names))):
            agent = aif.ActiveInferenceAgent(
                name=agent_names[i],
                role=role_names[i],
                gamma=1.0,
                alpha=16.0,
                learning_rate=0.1,
                health_prior='uncertain',
            )
            self.agents.append(agent)

        # Environment state (ground truth, hidden from agents)
        self.true_health = 0  # 0=healthy, 1=stressed, 2=declining
        self.true_urgency = 0  # 0=balanced, 1=bugs, 2=docs, 3=review

        # Stress schedule: stress hits at sprint 2 and 4
        self.stress_sprints = {2, 4} if num_sprints >= 3 else set()

        # Available tasks per sprint (not all types always available)
        self.available_tasks: list[str] = list(aif.ACTIONS[:4])

        # Metrics tracking
        self.history: list[dict[str, Any]] = []

    def _get_reward(self, agent: aif.ActiveInferenceAgent, action: str) -> float:
        """Compute reward for an action with stochastic success.

        Matches Vidhi's original: 70% base success, +3 preferred, +1 other,
        -1 failure, 0 skip. Stress degrades success probability.
        """
        if action == 'skip':
            return 0.0

        if action not in self.available_tasks:
            return -0.5  # Task type not available this sprint

        preferred = {
            'contributor': 'bug_fix',
            'innovator': 'feature',
            'knowledge_curator': 'documentation',
            'maintainer': 'code_review',
        }
        is_preferred = (action == preferred.get(agent.role, ''))

        # Success probability: 70% base, +15% if preferred, -20% if stressed
        success_prob = 0.70
        if is_preferred:
            success_prob += 0.15
        if self.true_health >= 2:  # declining
            success_prob -= 0.20
        elif self.true_health >= 1:  # stressed
            success_prob -= 0.10
        success_prob = np.clip(success_prob, 0.1, 0.95)

        succeeded = self.rng.random() < success_prob

        if not succeeded:
            return -1.0

        # Project need: actions matching urgency get bonus
        urgency_map = {1: 'bug_fix', 2: 'documentation', 3: 'code_review'}
        matches_need = (action == urgency_map.get(self.true_urgency, ''))

        if is_preferred:
            reward = 3.0
        elif matches_need:
            reward = 2.5  # Urgency bonus
        else:
            reward = 1.0

        return reward

    def _compute_prediction_error(
        self, agent: aif.ActiveInferenceAgent, observation: str
    ) -> float:
        """Level 1: Compute prediction error (surprise)."""
        if not self.level.use_prediction_error:
            return 0.0

        # Compare agent's belief about health with actual observation
        hi_idx = aif.HI_OBSERVATIONS.index(observation)
        predicted_obs = np.zeros(aif.NUM_HI_OBS)
        for h in range(aif.NUM_HEALTH):
            for u in range(aif.NUM_URGENCY):
                predicted_obs += (
                    agent.A[0][:, h, u]
                    * agent.beliefs[0][h]
                    * agent.beliefs[1][u]
                )
        predicted_obs = np.clip(predicted_obs / predicted_obs.sum(), 1e-16, None)

        # Surprise = -log P(observation | beliefs)
        surprise = -np.log(predicted_obs[hi_idx])
        return float(surprise)

    def _get_epistemic_bonus(
        self, agent: aif.ActiveInferenceAgent, action_idx: int
    ) -> float:
        """Level 2: Compute information gain for an action."""
        if not self.level.use_epistemic_value:
            return 0.0

        # Predict next state
        predicted_health = agent.B[0][:, :, action_idx] @ agent.beliefs[0]

        # Compute entropy of observations given predicted state
        entropy = 0.0
        for h in range(aif.NUM_HEALTH):
            for u in range(aif.NUM_URGENCY):
                obs_probs = np.clip(agent.A[0][:, h, u], 1e-16, None)
                state_prob = predicted_health[h] * agent.beliefs[1][u]
                entropy -= state_prob * np.sum(obs_probs * np.log(obs_probs))

        return float(entropy)  # Higher entropy = more to learn = higher epistemic value

    def _update_precision(
        self, agent: aif.ActiveInferenceAgent, prediction_error: float, sprint: int
    ) -> None:
        """Level 6: Adapt precision based on prediction error history."""
        if not self.level.use_precision_dynamics:
            return

        # Precision increases when prediction errors are low (confident)
        # Precision decreases when prediction errors are high (surprised)
        target_gamma = 1.0 / (1.0 + prediction_error)

        # Smooth update
        agent.gamma = 0.7 * agent.gamma + 0.3 * target_gamma

        # Also increase precision over time (exploitation ramp)
        time_factor = 1.0 + 0.1 * sprint
        agent.gamma *= time_factor
        agent.gamma = np.clip(agent.gamma, 0.1, 10.0)

    def _apply_stress_event(self, sprint: int) -> None:
        """Apply stress events at scheduled sprints."""
        if sprint not in self.stress_sprints:
            return
        # Stress: health degrades, urgency shifts to bugs
        self.true_health = min(2, self.true_health + 1)
        self.true_urgency = 1  # bugs become critical
        # Remove one task type (simulating resource constraint)
        scarce = self.rng.choice(['documentation', 'code_review'])
        if scarce in self.available_tasks:
            self.available_tasks.remove(scarce)

    def _simulate_environment_dynamics(self, actions: list[str]) -> None:
        """Update true hidden state based on collective actions."""
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        # Health requires active maintenance — needs both bug_fix AND code_review
        maintenance = (action_counts.get('bug_fix', 0)
                       + action_counts.get('code_review', 0))
        growth = action_counts.get('feature', 0)

        if maintenance >= 3:
            self.true_health = max(0, self.true_health - 1)  # Healing
        elif growth > maintenance:
            # Growing without maintaining = technical debt
            self.true_health = min(2, self.true_health + 1)

        # Urgency: whichever area is most neglected becomes urgent
        area_counts = {
            1: action_counts.get('bug_fix', 0),
            2: action_counts.get('documentation', 0),
            3: action_counts.get('code_review', 0),
        }
        # Find the most neglected area
        min_area = min(area_counts, key=area_counts.get)
        if area_counts[min_area] == 0:
            self.true_urgency = min_area
        else:
            self.true_urgency = 0  # balanced

        # Restore available tasks for next sprint (stress may remove again)
        self.available_tasks = list(aif.ACTIONS[:4])

    def _generate_observation(self) -> str:
        """Generate an HI observation from true state."""
        probs = aif.build_A_matrix()[0][:, self.true_health, self.true_urgency]
        probs = probs / probs.sum()
        obs_idx = self.rng.choice(aif.NUM_HI_OBS, p=probs)
        return aif.HI_OBSERVATIONS[obs_idx]

    def run_sprint(self, sprint_num: int) -> dict[str, Any]:
        """Run one sprint through the experiment level's pipeline."""
        # 0. Apply stress events before anything else
        self._apply_stress_event(sprint_num)

        sprint_data = {
            'sprint': sprint_num,
            'level': self.level.level,
            'level_name': self.level.name,
            'true_health': aif.PROJECT_HEALTH_STATES[self.true_health],
            'true_urgency': aif.TASK_URGENCY_STATES[self.true_urgency],
            'available_tasks': list(self.available_tasks),
            'is_stress_sprint': sprint_num in self.stress_sprints,
        }

        # 1. Observation
        hi_obs = self._generate_observation()
        task_obs = 'success'  # Simplified
        sprint_data['observation'] = hi_obs

        # 2. Belief updating (Level 3+)
        if self.level.use_belief_updating:
            for agent in self.agents:
                agent.observe(hi_obs, task_obs)

        # 3. Prediction error (Level 1+)
        prediction_errors = {}
        for agent in self.agents:
            pe = self._compute_prediction_error(agent, hi_obs)
            prediction_errors[agent.name] = pe
        sprint_data['prediction_errors'] = prediction_errors

        # 4. Action selection
        actions = {}
        action_probs = {}
        for agent in self.agents:
            if self.level.use_efe_policy:
                # Level 4+: EFE-based policy selection
                # Adjust alpha based on level features for differentiation
                orig_alpha = agent.alpha
                if self.level.use_precision_dynamics:
                    # L6+: precision-modulated exploration
                    agent.alpha = agent.gamma * 4.0
                elif self.level.use_habit_learning:
                    # L5: habits soften policy (more diverse actions)
                    agent.alpha = 8.0
                else:
                    # L4: base EFE
                    agent.alpha = 4.0
                action, probs = agent.decide()  # returns (action_string, probs)
                agent.alpha = orig_alpha
            elif self.level.use_epistemic_value:
                # Level 2-3: Reward + epistemic bonus with softmax sampling
                scores = np.zeros(aif.NUM_ACTIONS)
                for a_idx, a_name in enumerate(aif.ACTIONS):
                    reward_est = self._get_reward(agent, a_name)
                    epist = self._get_epistemic_bonus(agent, a_idx)
                    scores[a_idx] = reward_est + 0.5 * epist
                # Softmax sample (not greedy) — epistemic value drives exploration
                temperature = 0.5 if self.level.use_belief_updating else 1.0
                probs = aif._softmax(scores * (1.0 / max(temperature, 0.1)))
                action_idx = self.rng.choice(aif.NUM_ACTIONS, p=probs)
                action = aif.ACTIONS[action_idx]
            elif self.level.use_prediction_error:
                # Level 1: Greedy + epsilon exploration driven by surprise
                scores = np.zeros(aif.NUM_ACTIONS)
                for a_idx, a_name in enumerate(aif.ACTIONS):
                    scores[a_idx] = self._get_reward(agent, a_name)
                pe = prediction_errors.get(agent.name, 0.0)
                # High surprise → more exploration (higher epsilon)
                epsilon = min(0.4, 0.05 + 0.1 * pe)
                if self.rng.random() < epsilon:
                    probs = np.ones(aif.NUM_ACTIONS) / aif.NUM_ACTIONS
                    action_idx = self.rng.choice(aif.NUM_ACTIONS, p=probs)
                    action = aif.ACTIONS[action_idx]
                else:
                    action = aif.ACTIONS[int(np.argmax(scores))]
                    probs = aif._softmax(scores)
            else:
                # Level 0: Pure reward-based (greedy)
                scores = np.zeros(aif.NUM_ACTIONS)
                for a_idx, a_name in enumerate(aif.ACTIONS):
                    scores[a_idx] = self._get_reward(agent, a_name)
                action = aif.ACTIONS[int(np.argmax(scores))]
                probs = aif._softmax(scores)

            actions[agent.name] = action
            action_probs[agent.name] = {
                aif.ACTIONS[i]: float(probs[i]) for i in range(aif.NUM_ACTIONS)
            }

        sprint_data['actions'] = actions
        sprint_data['action_probabilities'] = action_probs

        # 5. Environment dynamics
        self._simulate_environment_dynamics(list(actions.values()))

        # 6. Reward computation
        rewards = {}
        for agent in self.agents:
            action = actions[agent.name]
            reward = self._get_reward(agent, action)
            rewards[agent.name] = reward
        sprint_data['rewards'] = rewards

        # 7. Habit learning (Level 5+)
        if self.level.use_habit_learning:
            for agent in self.agents:
                reward = rewards[agent.name]
                valence = (reward - 1.0) / 2.0  # Normalize to [-0.5, 1.0]
                agent.learn(valence)
        sprint_data['habits'] = {
            agent.name: {
                aif.ACTIONS[i]: float(agent.E[i])
                for i in range(aif.NUM_ACTIONS)
            }
            for agent in self.agents
        }

        # 8. Precision dynamics (Level 6+)
        for agent in self.agents:
            pe = prediction_errors[agent.name]
            self._update_precision(agent, pe, sprint_num)
        sprint_data['precisions'] = {
            agent.name: float(agent.gamma) for agent in self.agents
        }

        # 9. Compute HI-like metric
        total_reward = sum(rewards.values())
        max_reward = len(self.agents) * 3.0
        hi = total_reward / max_reward if max_reward > 0 else 0.0

        # Diversity: how many different action types were chosen?
        unique_actions = len(set(a for a in actions.values() if a != 'skip'))
        diversity = unique_actions / len(social_data.TASK_TYPES)

        # Coverage: were all task types addressed?
        task_types_covered = set()
        for a in actions.values():
            if a in social_data.TASK_TYPES[:4]:  # bug_fix, feature, documentation, code_review
                task_types_covered.add(a)
        coverage = len(task_types_covered) / len(social_data.TASK_TYPES[:4])

        sprint_data['harmony_index'] = hi
        sprint_data['diversity'] = diversity
        sprint_data['coverage'] = coverage

        # Agent belief summaries
        sprint_data['beliefs'] = {}
        for agent in self.agents:
            state = agent.get_state_summary()
            sprint_data['beliefs'][agent.name] = {
                'health': state['beliefs_health'],
                'urgency': state['beliefs_urgency'],
            }

        return sprint_data

    def run(self) -> dict[str, Any]:
        """Run the full experiment."""
        t0 = time.time()

        for sprint in range(self.num_sprints):
            sprint_data = self.run_sprint(sprint)
            self.history.append(sprint_data)

        duration = time.time() - t0

        # Aggregate metrics
        hi_values = [s['harmony_index'] for s in self.history]
        coverage_values = [s['coverage'] for s in self.history]
        diversity_values = [s['diversity'] for s in self.history]

        # Strategy change: did agents change actions across sprints?
        strategy_changes = 0
        for agent in self.agents:
            if len(agent.action_history) >= 2:
                if len(set(agent.action_history)) > 1:
                    strategy_changes += 1
        strategy_diversity = strategy_changes / len(self.agents)

        # Resilience Quotient (RQ) calculation
        # RQ = mean(HI_after_stress) / mean(HI_before_stress)
        rq = 1.0
        if self.stress_sprints:
            first_stress = min(self.stress_sprints)
            hi_before = hi_values[:first_stress]
            hi_after = hi_values[first_stress:]
            if hi_before and hi_after:
                mean_before = np.mean(hi_before)
                mean_after = np.mean(hi_after)
                if mean_before > 0:
                    rq = float(mean_after / mean_before)

        return {
            'level': self.level.level,
            'level_name': self.level.name,
            'description': self.level.description,
            'new_concept': self.level.new_concept,
            'rl_analog': self.level.rl_analog,
            'aif_mechanism': self.level.aif_mechanism,
            'num_sprints': self.num_sprints,
            'duration_s': duration,
            'mean_hi': float(np.mean(hi_values)),
            'final_hi': hi_values[-1] if hi_values else 0.0,
            'resilience_quotient': rq,
            'mean_coverage': float(np.mean(coverage_values)),
            'mean_diversity': float(np.mean(diversity_values)),
            'strategy_diversity': strategy_diversity,
            'hi_trajectory': hi_values,
            'sprint_history': self.history,
        }


# =============================================================================
# Cross-System Comparison Experiments
# =============================================================================

@dataclasses.dataclass
class CrossSystemExperiment:
    """Configuration for a cross-system comparison experiment."""
    name: str
    description: str
    comparison_target: str  # "rohira", "llamosc", or "ostrom"

    # Simulation parameters
    num_agents: int
    num_sprints: int
    num_seeds: int
    stress_mode: str  # "dropout_only", "all", "none"
    governance: str  # "free_choice", "dictator", "meritocratic"

    # AIF configuration
    use_active_inference: bool
    experiment_level: int | None  # None = use full Concordia sim, 0-7 = use experiment ladder

    # Flags
    requires_llm: bool = True
    estimated_hours_per_seed: float = 1.0


CROSS_SYSTEM_EXPERIMENTS = [
    CrossSystemExperiment(
        name="A1",
        description="Match Rohira's setup: LLM-only agents, dropout stress, 10 agents",
        comparison_target="rohira",
        num_agents=10, num_sprints=10, num_seeds=5,
        stress_mode="dropout_only",
        governance="free_choice",
        use_active_inference=False,
        experiment_level=None,
        requires_llm=True,
        estimated_hours_per_seed=1.0,
    ),
    CrossSystemExperiment(
        name="A2",
        description="AIF contribution: LLM+AIF agents vs A1 LLM-only",
        comparison_target="rohira",
        num_agents=10, num_sprints=10, num_seeds=5,
        stress_mode="dropout_only",
        governance="free_choice",
        use_active_inference=True,
        experiment_level=None,
        requires_llm=True,
        estimated_hours_per_seed=1.0,
    ),
    CrossSystemExperiment(
        name="A3",
        description="Convergence comparison: AIF-only (Level 4), 50 sprints, no LLM",
        comparison_target="rohira",
        num_agents=10, num_sprints=50, num_seeds=10,
        stress_mode="dropout_only",
        governance="free_choice",
        use_active_inference=True,
        experiment_level=4,
        requires_llm=False,
        estimated_hours_per_seed=0.01,
    ),
    CrossSystemExperiment(
        name="B1",
        description="Governance baseline: free-choice with all stress types",
        comparison_target="llamosc",
        num_agents=8, num_sprints=5, num_seeds=5,
        stress_mode="all",
        governance="free_choice",
        use_active_inference=True,
        experiment_level=None,
        requires_llm=True,
        estimated_hours_per_seed=0.5,
    ),
    CrossSystemExperiment(
        name="B2",
        description="Dictator governance: vs LLAMOSC authoritarian mode",
        comparison_target="llamosc",
        num_agents=8, num_sprints=5, num_seeds=5,
        stress_mode="all",
        governance="dictator",
        use_active_inference=True,
        experiment_level=None,
        requires_llm=True,
        estimated_hours_per_seed=0.5,
    ),
    CrossSystemExperiment(
        name="B3",
        description="Meritocratic governance: vs LLAMOSC merit-based mode",
        comparison_target="llamosc",
        num_agents=8, num_sprints=5, num_seeds=5,
        stress_mode="all",
        governance="meritocratic",
        use_active_inference=True,
        experiment_level=None,
        requires_llm=True,
        estimated_hours_per_seed=0.5,
    ),
    CrossSystemExperiment(
        name="C1",
        description="Ostrom contribution: LLM+AIF with Ostrom-aligned priors",
        comparison_target="ostrom",
        num_agents=10, num_sprints=10, num_seeds=5,
        stress_mode="dropout_only",
        governance="free_choice",
        use_active_inference=True,
        experiment_level=None,
        requires_llm=True,
        estimated_hours_per_seed=1.0,
    ),
]


def run_cross_system_experiment(
    experiment: CrossSystemExperiment,
    seed: int,
    vllm_url: str | None = None,
    model_name: str | None = None,
    use_mock: bool = False,
    nvidia_nim: bool = False,
    output_dir: str = '/tmp/sustain_hub_cross_system',
) -> dict[str, Any]:
    """Run a single seed of a cross-system comparison experiment.

    For experiment_level != None, uses the standalone ExperimentRunner.
    For experiment_level == None, shells out to the full Concordia simulation.

    Returns a results dict with metrics.
    """
    import subprocess
    import sys

    exp_dir = os.path.join(output_dir, experiment.name, f'seed_{seed}')
    os.makedirs(exp_dir, exist_ok=True)

    if experiment.experiment_level is not None:
        # Use the standalone AIF experiment runner (no LLM needed)
        level = EXPERIMENT_LEVELS[experiment.experiment_level]
        runner = ExperimentRunner(
            level=level,
            num_agents=experiment.num_agents,
            num_sprints=experiment.num_sprints,
            seed=seed,
        )
        result = runner.run()
        # Save result
        result['experiment'] = experiment.name
        result['seed'] = seed
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2)
        return result
    else:
        # Use the full Concordia simulation via subprocess
        cmd = [
            sys.executable, '-m', 'examples.games.sustain_hub.run',
            f'--num_sprints={experiment.num_sprints}',
            f'--community_size={experiment.num_agents}',
            f'--seed={seed}',
            f'--output_dir={exp_dir}',
            '--skip_backstory',
            '--fast',
        ]
        if not experiment.use_active_inference:
            cmd.append('--nouse_active_inference')
        if experiment.governance != 'free_choice':
            cmd.append(f'--governance={experiment.governance}')
        if experiment.stress_mode == 'none':
            cmd.append('--noenable_stress')
        if use_mock:
            cmd.append('--use_mock')
        elif nvidia_nim:
            cmd.append('--nvidia_nim')
            if model_name:
                cmd.append(f'--model_name={model_name}')
        elif vllm_url:
            cmd.extend([f'--vllm_url={vllm_url}'])
            if model_name:
                cmd.append(f'--model_name={model_name}')

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        results_path = os.path.join(exp_dir, 'results.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                return json.load(f)
        else:
            return {'status': 'crash', 'stderr': result.stderr[-500:] if result.stderr else ''}


def run_cross_system_suite(spec: str):
    """Run a suite of cross-system comparison experiments."""
    if spec == 'all':
        experiments = CROSS_SYSTEM_EXPERIMENTS
    elif spec in ('rohira', 'llamosc', 'ostrom'):
        experiments = [e for e in CROSS_SYSTEM_EXPERIMENTS if e.comparison_target == spec]
    else:
        names = [n.strip() for n in spec.split(',')]
        experiments = [e for e in CROSS_SYSTEM_EXPERIMENTS if e.name in names]

    if not experiments:
        print(f'No experiments matching: {spec}')
        return

    print(f'Running {len(experiments)} cross-system experiments:')
    for exp in experiments:
        print(f'  {exp.name}: {exp.description}')

    all_results = {}
    for exp in experiments:
        print(f'\n{"="*60}')
        print(f'Experiment {exp.name}: {exp.description}')
        print(f'  {exp.num_agents} agents, {exp.num_sprints} sprints, '
              f'{exp.num_seeds} seeds, governance={exp.governance}')
        print(f'{"="*60}')

        exp_results = []
        for seed in range(exp.num_seeds):
            print(f'  Seed {seed+1}/{exp.num_seeds}...', end=' ', flush=True)
            t0 = time.time()
            result = run_cross_system_experiment(
                exp, seed=seed,
                vllm_url=FLAGS.vllm_url,
                model_name=FLAGS.model_name,
                use_mock=FLAGS.use_mock,
                nvidia_nim=FLAGS.nvidia_nim,
            )
            duration = time.time() - t0
            hi = result.get('harmony_index', result.get('mean_hi', 0.0))
            print(f'HI={hi:.3f} ({duration:.0f}s)')
            exp_results.append(result)

        all_results[exp.name] = exp_results

    # Print comparison table
    print(f'\n{"="*60}')
    print('CROSS-SYSTEM COMPARISON RESULTS')
    print(f'{"="*60}')
    print(f'{"Exp":<5} {"Description":<45} {"Mean HI":<10} {"Mean RQ":<10} {"Seeds":<6}')
    print('-' * 76)
    for exp in experiments:
        results = all_results.get(exp.name, [])
        if results:
            mean_hi = sum(r.get('harmony_index', r.get('mean_hi', 0.0)) for r in results) / len(results)
            mean_rq = sum(r.get('resilience_quotient', 0.0) for r in results) / len(results)
            print(f'{exp.name:<5} {exp.description[:45]:<45} {mean_hi:<10.4f} {mean_rq:<10.4f} {len(results):<6}')

    # Save all results
    output_path = os.path.join(FLAGS.output_dir, 'cross_system_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Filter non-serializable values
    serializable = {}
    for name, results in all_results.items():
        serializable[name] = [
            {k: v for k, v in r.items() if k != 'structured_log'}
            for r in results
        ]
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f'\nAll results saved to {output_path}')


# =============================================================================
# Main
# =============================================================================

def print_comparison_table(results: list[dict]) -> None:
    """Print a comparison table across experiment levels."""
    print("\n" + "=" * 100)
    print("EXPERIMENT LADDER: RL → ACTIVE INFERENCE")
    print("=" * 100)
    print(f"{'Lvl':<4} {'Name':<30} {'Mean HI':<10} {'Coverage':<10} "
          f"{'Diversity':<10} {'Strat Δ':<10} {'New Concept':<30}")
    print("-" * 100)

    for r in results:
        print(f"{r['level']:<4} {r['level_name']:<30} {r['mean_hi']:<10.4f} "
              f"{r['mean_coverage']:<10.4f} {r['mean_diversity']:<10.4f} "
              f"{r['strategy_diversity']:<10.4f} {r['new_concept']:<30}")

    print("=" * 100)

    # Show the conceptual mapping
    print("\nConceptual Bridge: RL → Active Inference")
    print("-" * 80)
    for r in results:
        print(f"\nLevel {r['level']}: {r['level_name']}")
        print(f"  RL analog:       {r['rl_analog']}")
        print(f"  AIF mechanism:   {r['aif_mechanism']}")
        hi_traj = " → ".join(f"{h:.3f}" for h in r['hi_trajectory'])
        print(f"  HI trajectory:   {hi_traj}")


def main(argv):
    del argv

    if FLAGS.cross_system:
        run_cross_system_suite(FLAGS.cross_system)
        return

    # Parse level specification
    if FLAGS.level.lower() == 'all':
        levels_to_run = list(range(len(EXPERIMENT_LEVELS)))
    else:
        levels_to_run = [int(x.strip()) for x in FLAGS.level.split(',')]

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    results = []
    for level_idx in levels_to_run:
        if level_idx >= len(EXPERIMENT_LEVELS):
            print(f"Warning: Level {level_idx} not defined, skipping.")
            continue

        level = EXPERIMENT_LEVELS[level_idx]
        print(f"\n{'='*60}")
        print(f"Level {level.level}: {level.name}")
        print(f"New concept: {level.new_concept}")
        print(f"{'='*60}")

        runner = ExperimentRunner(
            level=level,
            num_agents=8,
            num_sprints=FLAGS.num_sprints,
            seed=FLAGS.seed,
        )
        result = runner.run()
        results.append(result)

        print(f"  Mean HI:     {result['mean_hi']:.4f}")
        print(f"  Coverage:    {result['mean_coverage']:.4f}")
        print(f"  Diversity:   {result['mean_diversity']:.4f}")
        print(f"  Strategy Δ:  {result['strategy_diversity']:.4f}")
        print(f"  Duration:    {result['duration_s']:.2f}s")

        # Save individual result
        result_file = os.path.join(FLAGS.output_dir, f'level_{level_idx}.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    # Print comparison table
    if len(results) > 1:
        print_comparison_table(results)

    # Save combined results
    combined_file = os.path.join(FLAGS.output_dir, 'ladder_results.json')
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {FLAGS.output_dir}/")


# =============================================================================
# Cross-System Comparison Experiments (Step 5)
# =============================================================================

COMPARISON_EXPERIMENTS = {
    # Series A: Match Rohira's setup
    "A1": {
        "name": "Concordia LLM-only vs Rohira",
        "community_size": 10,
        "num_sprints": 10,
        "stress_types": ["contributor_dropout"],
        "governance": "free_choice",
        "use_active_inference": False,
        "seeds": 5,
        "purpose": "Match Rohira's setup — does LLM reasoning help or hurt?",
    },
    "A2": {
        "name": "Concordia LLM+AIF vs Rohira",
        "community_size": 10,
        "num_sprints": 10,
        "stress_types": ["contributor_dropout"],
        "governance": "free_choice",
        "use_active_inference": True,
        "seeds": 5,
        "purpose": "AIF contribution — does Bayesian guidance improve LLM decisions?",
    },
    "A3": {
        "name": "Concordia AIF-only convergence",
        "community_size": 10,
        "num_sprints": 50,
        "stress_types": ["contributor_dropout"],
        "governance": "free_choice",
        "use_active_inference": True,
        "llm_level": 4,  # AIF Level 4 only, no LLM
        "seeds": 10,
        "purpose": "Convergence comparison — same math, different framework",
    },
    # Series B: Governance comparison (vs LLAMOSC)
    "B1": {
        "name": "Free choice governance",
        "community_size": 8,
        "num_sprints": 5,
        "stress_types": None,  # all stress types
        "governance": "free_choice",
        "use_active_inference": True,
        "seeds": 5,
        "purpose": "Governance baseline",
    },
    "B2": {
        "name": "Dictator governance",
        "community_size": 8,
        "num_sprints": 5,
        "stress_types": None,
        "governance": "dictator",
        "use_active_inference": True,
        "seeds": 5,
        "purpose": "vs LLAMOSC dictator — same governance, richer cognition",
    },
    "B3": {
        "name": "Meritocratic governance",
        "community_size": 8,
        "num_sprints": 5,
        "stress_types": None,
        "governance": "meritocratic",
        "use_active_inference": True,
        "seeds": 5,
        "purpose": "vs LLAMOSC meritocratic — same governance, richer cognition",
    },
    # Series C: Ostrom contribution
    "C1": {
        "name": "LLM+AIF+Ostrom priors",
        "community_size": 10,
        "num_sprints": 10,
        "stress_types": ["contributor_dropout"],
        "governance": "free_choice",
        "use_active_inference": True,
        "use_ostrom": True,
        "seeds": 5,
        "purpose": "Ostrom priors as Bayesian formalization of governance principles",
    },
}


def run_comparison_experiment(
    experiment_key: str,
    vllm_url: str,
    model_name: str,
    output_dir: str = '/tmp/sustain_hub_comparison',
) -> dict[str, Any]:
    """Run a single cross-system comparison experiment across seeds.

    Args:
        experiment_key: Key into COMPARISON_EXPERIMENTS (e.g. "A1").
        vllm_url: vLLM API base URL.
        model_name: LLM model name.
        output_dir: Root output directory.

    Returns:
        Summary dict with mean/std for each metric across seeds.
    """
    import subprocess
    import sys

    config = COMPARISON_EXPERIMENTS[experiment_key]
    num_seeds = config['seeds']
    seed_results: list[dict[str, Any]] = []

    print(f'\n{"=" * 60}')
    print(f'Experiment {experiment_key}: {config["name"]}')
    print(f'  Purpose: {config["purpose"]}')
    print(f'  Seeds: {num_seeds}  |  Sprints: {config["num_sprints"]}  '
          f'|  Community: {config["community_size"]}')
    print(f'{"=" * 60}')

    # Check if this is an AIF-only experiment (uses the experiment ladder
    # runner instead of the full simulation via run.py)
    is_aif_only = 'llm_level' in config

    for seed_idx in range(num_seeds):
        seed_val = 42 + seed_idx
        run_dir = os.path.join(
            output_dir, experiment_key, f'seed_{seed_idx}'
        )
        os.makedirs(run_dir, exist_ok=True)

        if is_aif_only:
            # Run via the experiment ladder (AIF-only, no LLM subprocess)
            level_idx = config['llm_level']
            level = EXPERIMENT_LEVELS[level_idx]
            runner = ExperimentRunner(
                level=level,
                num_agents=min(config['community_size'], 6),
                num_sprints=config['num_sprints'],
                seed=seed_val,
                enable_stress=bool(config.get('stress_types')),
            )
            if config.get('use_ostrom'):
                apply_ostrom_to_runner(runner)
            ladder_result = runner.run()
            adapted = ladder_results_to_evaluate_schema(ladder_result)

            # Compute metrics using evaluate.py functions
            # Lazy-import to avoid absl flag conflicts at module load time
            for _flag_name in list(FLAGS):
                FLAGS[_flag_name].allow_override = True
            from examples.games.sustain_hub import evaluate as _eval_mod

            metrics = _eval_mod.compute_sustain_score(adapted)
            metrics['status'] = 'ok'

            # Save per-seed results
            results_path = os.path.join(run_dir, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(adapted, f, indent=2, default=str)

        else:
            # Run via subprocess to run.py (full Concordia simulation)
            cmd = [
                sys.executable, '-m', 'examples.games.sustain_hub.run',
                f'--num_sprints={config["num_sprints"]}',
                f'--community_size={config["community_size"]}',
                f'--governance={config["governance"]}',
                f'--seed={seed_val}',
                '--skip_backstory',
                '--fast',
                f'--output_dir={run_dir}',
                f'--vllm_url={vllm_url}',
                f'--model_name={model_name}',
            ]
            if config['use_active_inference']:
                cmd.append('--use_active_inference')
            else:
                cmd.append('--nouse_active_inference')
            if config.get('stress_types'):
                cmd.append(f'--stress_types={",".join(config["stress_types"])}')
                cmd.append('--enable_stress')
            else:
                # None means all stress types
                cmd.append('--enable_stress')

            print(f'  Seed {seed_idx} (seed={seed_val})...', end=' ',
                  flush=True)
            t0 = time.time()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800,
            )
            duration = time.time() - t0

            if result.returncode != 0:
                print(f'CRASHED ({duration:.0f}s)')
                stderr_tail = result.stderr[-300:] if result.stderr else ''
                if stderr_tail:
                    print(f'    stderr: {stderr_tail}')
                seed_results.append({'status': 'crash'})
                continue

            # Load results
            results_path = os.path.join(run_dir, 'results.json')
            if not os.path.exists(results_path):
                print(f'no results.json ({duration:.0f}s)')
                seed_results.append({'status': 'crash'})
                continue

            with open(results_path) as f:
                data = json.load(f)

            for _flag_name in list(FLAGS):
                FLAGS[_flag_name].allow_override = True
            from examples.games.sustain_hub import evaluate as _eval_mod

            metrics = _eval_mod.compute_sustain_score(data)
            metrics['status'] = 'ok'
            print(f'HI={metrics["harmony_index"]:.3f} '
                  f'CHS={metrics["chs"]:.3f} ({duration:.0f}s)')

        seed_results.append(metrics)

    # Aggregate across seeds
    ok_results = [r for r in seed_results if r.get('status') == 'ok']
    if not ok_results:
        print(f'  WARNING: All seeds crashed for {experiment_key}')
        return {
            'key': experiment_key,
            'name': config['name'],
            'purpose': config['purpose'],
            'status': 'all_crashed',
            'num_ok': 0,
            'num_seeds': num_seeds,
        }

    metric_keys = [
        'harmony_index', 'resilience_quotient', 'mean_brs', 'sue', 'chs',
    ]
    summary: dict[str, Any] = {
        'key': experiment_key,
        'name': config['name'],
        'purpose': config['purpose'],
        'config': config,
        'status': 'ok',
        'num_ok': len(ok_results),
        'num_seeds': num_seeds,
    }
    for mk in metric_keys:
        values = [r[mk] for r in ok_results if mk in r]
        if values:
            summary[f'{mk}_mean'] = float(np.mean(values))
            summary[f'{mk}_std'] = float(np.std(values))
        else:
            summary[f'{mk}_mean'] = 0.0
            summary[f'{mk}_std'] = 0.0

    return summary


def run_comparison_suite(
    experiment_keys: list[str] | str = 'all',
    vllm_url: str = '',
    model_name: str = 'Qwen/Qwen2.5-7B-Instruct',
    output_dir: str = '/tmp/sustain_hub_comparison',
) -> dict[str, Any]:
    """Run the full cross-system comparison suite.

    Args:
        experiment_keys: List of keys (e.g. ["A1","A2"]) or "all".
        vllm_url: vLLM API base URL.
        model_name: LLM model name.
        output_dir: Root output directory.

    Returns:
        Dict with all experiment results keyed by experiment key.
    """
    if experiment_keys == 'all' or experiment_keys == ['all']:
        keys = list(COMPARISON_EXPERIMENTS.keys())
    elif isinstance(experiment_keys, str):
        keys = [k.strip() for k in experiment_keys.split(',')]
    else:
        keys = list(experiment_keys)

    # Validate keys
    for k in keys:
        if k not in COMPARISON_EXPERIMENTS:
            raise ValueError(
                f'Unknown experiment key: {k}. '
                f'Valid keys: {list(COMPARISON_EXPERIMENTS.keys())}'
            )

    os.makedirs(output_dir, exist_ok=True)

    all_results: dict[str, Any] = {}
    for key in keys:
        result = run_comparison_experiment(
            key, vllm_url, model_name, output_dir,
        )
        all_results[key] = result

    # Print formatted table
    format_comparison_table(all_results)

    # Save combined results
    comparison_file = os.path.join(output_dir, 'comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nComparison results saved to {comparison_file}')

    return all_results


def format_comparison_table(results: dict[str, Any]) -> None:
    """Print a formatted comparison table across experiments.

    Args:
        results: Dict mapping experiment key -> summary dict from
            run_comparison_experiment.
    """
    print(f'\n{"=" * 120}')
    print('CROSS-SYSTEM COMPARISON (Step 5)')
    print(f'{"=" * 120}')

    header = (
        f'{"Exp":<5} '
        f'{"HI (mean+/-std)":<18} '
        f'{"RQ":<18} '
        f'{"BRS":<18} '
        f'{"SUE":<18} '
        f'{"CHS":<18} '
        f'{"Purpose"}'
    )
    print(header)
    print('-' * 120)

    for key in sorted(results.keys()):
        r = results[key]
        if r.get('status') != 'ok':
            print(f'{key:<5} {"CRASHED":<18} {"":18} {"":18} {"":18} '
                  f'{"":18} {r.get("purpose", "")}')
            continue

        def _fmt(metric_name: str) -> str:
            mean = r.get(f'{metric_name}_mean', 0.0)
            std = r.get(f'{metric_name}_std', 0.0)
            return f'{mean:.3f} +/- {std:.3f}'

        print(
            f'{key:<5} '
            f'{_fmt("harmony_index"):<18} '
            f'{_fmt("resilience_quotient"):<18} '
            f'{_fmt("mean_brs"):<18} '
            f'{_fmt("sue"):<18} '
            f'{_fmt("chs"):<18} '
            f'{r.get("purpose", "")}'
        )

    print(f'{"=" * 120}')

    # Summary statistics
    ok_results = {k: v for k, v in results.items() if v.get('status') == 'ok'}
    if ok_results:
        seeds_ok = sum(v['num_ok'] for v in ok_results.values())
        seeds_total = sum(v['num_seeds'] for v in ok_results.values())
        print(f'\nExperiments completed: {len(ok_results)}/{len(results)}')
        print(f'Seeds completed: {seeds_ok}/{seeds_total}')


# =============================================================================
# CLI flags for comparison mode
# =============================================================================

flags.DEFINE_bool(
    'comparison', False,
    'Run the cross-system comparison suite instead of the experiment ladder.')
flags.DEFINE_list(
    'experiments', None,
    'Comma-separated experiment keys for comparison mode '
    '(e.g. A1,A2,B1). Default: all.')


def _run_comparison_main() -> None:
    """Entry point for --comparison mode."""
    experiment_keys = FLAGS.experiments if FLAGS.experiments else 'all'
    vllm_url = FLAGS.vllm_url or ''
    model_name = FLAGS.model_name

    run_comparison_suite(
        experiment_keys=experiment_keys,
        vllm_url=vllm_url,
        model_name=model_name,
        output_dir=FLAGS.output_dir,
    )


# Patch main to handle --comparison flag
_original_main = main


def main(argv):
    if FLAGS.comparison:
        _run_comparison_main()
    else:
        _original_main(argv)


if __name__ == '__main__':
    app.run(main)
