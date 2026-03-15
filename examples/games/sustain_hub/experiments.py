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

"""Experiment ladder: RL -> Active Inference progression.

Each level adds one more active inference concept on top of the previous.
Run all levels to see how each concept contributes to agent behavior.

Usage:
  # Run a single level:
  python -m examples.games.sustain_hub.experiments --level=0

  # Run the full ladder:
  python -m examples.games.sustain_hub.experiments --level=all

  # Run levels 0-3 only:
  python -m examples.games.sustain_hub.experiments --level=0,1,2,3

  # Run Level 7 with a mock LLM (for testing):
  python -m examples.games.sustain_hub.experiments --level=7 --use_mock

  # Run Level 7 with a real vLLM backend:
  python -m examples.games.sustain_hub.experiments --level=7 --vllm_url=http://localhost:8000/v1

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

flags.DEFINE_string(
    'level', '0',
    'Experiment level(s): 0-7, "all", or comma-separated.')
flags.DEFINE_string(
    'output_dir', '/tmp/sustain_hub_experiments', 'Output directory.')
flags.DEFINE_integer('num_sprints', 3, 'Number of sprints per experiment.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_bool(
    'compute_sustain_score', False,
    'Compute SustainScore for each level after running.')
flags.DEFINE_string(
    'vllm_url', None,
    'vLLM API base URL for Level 7 (e.g. http://localhost:8000/v1).')
flags.DEFINE_string(
    'model_name', 'Qwen/Qwen2.5-7B-Instruct',
    'LLM model name for Level 7.')
flags.DEFINE_bool(
    'use_mock', False,
    'Use a mock LLM for testing Level 7 without a real model.')
flags.DEFINE_bool(
    'enable_stress', False,
    'Enable stress events (contributor dropout at sprint 2, security crisis '
    'at sprint 4) to test resilience.')
flags.DEFINE_bool(
    'ostrom', False,
    'Run Ostrom comparison: Level 4 with vs. without Ostrom priors.')
flags.DEFINE_string(
    'ostrom_principles', None,
    'Comma-separated Ostrom principles to test (default: all 8).')


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

    # Feature flags (cumulative -- each level adds to the previous)
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
        name='Pure RL Baseline',
        description=(
            'Standard reward-driven task selection. Agents choose based on '
            'immediate reward expectations (preferred task = +3, other = +1).'),
        new_concept='Reward signal',
        rl_analog='Q-learning / SARSA with fixed policy',
        aif_mechanism='None -- this is the RL baseline',
        use_reward_signals=True,
    ),
    ExperimentLevel(
        level=1,
        name='+ Prediction Error',
        description=(
            "Agents now track 'surprise' -- the difference between expected "
            'and observed outcomes. High surprise signals model mismatch.'),
        new_concept='Prediction error (surprise)',
        rl_analog="TD error d = r + gV(s') - V(s)",
        aif_mechanism='Free energy F = E_Q[ln Q(s) - ln P(o,s)] measures surprise',
        use_reward_signals=True,
        use_prediction_error=True,
    ),
    ExperimentLevel(
        level=2,
        name='+ Epistemic Value',
        description=(
            'Agents value actions that reduce uncertainty, not just actions '
            'that yield reward. An agent might choose an unfamiliar task to '
            "'learn' about that area of the project."),
        new_concept='Information gain (epistemic foraging)',
        rl_analog='e-greedy exploration (crude) or UCB (principled)',
        aif_mechanism="Epistemic value = -E_Q[H[P(o'|s')]] = expected info gain",
        use_reward_signals=True,
        use_prediction_error=True,
        use_epistemic_value=True,
    ),
    ExperimentLevel(
        level=3,
        name='+ Belief Updating',
        description=(
            'Agents maintain probabilistic beliefs about hidden project state '
            '(health, urgency) and update them via variational inference after '
            'each observation.'),
        new_concept='Variational inference / belief updating',
        rl_analog='State estimation (Kalman filter, particle filter)',
        aif_mechanism='Q(s) ~ P(o|s) x P(s) via iterative message passing',
        use_reward_signals=True,
        use_prediction_error=True,
        use_epistemic_value=True,
        use_belief_updating=True,
    ),
    ExperimentLevel(
        level=4,
        name='+ Expected Free Energy Policy',
        description=(
            'Actions selected by minimizing Expected Free Energy: a single '
            'objective that naturally balances reward-seeking (pragmatic) and '
            'information-seeking (epistemic).'),
        new_concept='Expected Free Energy (EFE) for policy selection',
        rl_analog='Q-value Q(s,a) -> negative EFE -G(pi)',
        aif_mechanism='G(pi) = -pragmatic - epistemic; P(pi) = s(-G + ln E)',
        use_reward_signals=True,
        use_prediction_error=True,
        use_epistemic_value=True,
        use_belief_updating=True,
        use_efe_policy=True,
    ),
    ExperimentLevel(
        level=5,
        name='+ Habit Learning',
        description=(
            'Agents accumulate Dirichlet concentration parameters for their '
            'policy prior (E-vector). Good actions become habitual across '
            'sprints. This is where Q-values map onto active inference.'),
        new_concept='Dirichlet learning / habit formation',
        rl_analog='Q-value accumulation across episodes',
        aif_mechanism='E(a) += eta x outcome; P(pi) = s(-G + ln E)',
        use_reward_signals=True,
        use_prediction_error=True,
        use_epistemic_value=True,
        use_belief_updating=True,
        use_efe_policy=True,
        use_habit_learning=True,
    ),
    ExperimentLevel(
        level=6,
        name='+ Precision Dynamics',
        description=(
            'The precision parameter gamma (inverse temperature) adapts over '
            'time. Early sprints: low precision -> more exploration. Later '
            'sprints: high precision -> more exploitation. Stress events '
            'collapse precision.'),
        new_concept='Precision weighting / adaptive confidence',
        rl_analog='Temperature annealing in softmax policy',
        aif_mechanism='gamma adapts based on prediction error history',
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
        name='+ LLM-as-Node (Full Hybrid)',
        description=(
            'LLM participates as a probabilistic node in the factor graph. '
            'LLMPrior generates informed priors from backstory. LLMObservation '
            'maps sprint narratives to state estimates. Bayesian inference '
            'combines them. (RxInfer.jl pattern in Python.)'),
        new_concept='LLM as probabilistic inference node',
        rl_analog='No RL analog -- this is beyond RL',
        aif_mechanism='LLM -> distribution -> factor graph -> VMP -> policy',
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
    - Each sprint: observe HI -> decide action -> receive outcome -> learn
    - Track metrics across sprints
    """

    def __init__(
        self,
        level: ExperimentLevel,
        num_agents: int = 6,
        num_sprints: int = 3,
        seed: int = 42,
        llm_model: Any = None,
        enable_stress: bool = False,
    ):
        self.level = level
        self.num_sprints = num_sprints
        self.enable_stress = enable_stress
        self.rng = np.random.RandomState(seed)

        # LLM model for Level 7 (None for levels 0-6)
        self.llm_model = llm_model
        self._use_llm = level.use_llm_nodes and llm_model is not None
        if level.use_llm_nodes and llm_model is None:
            print(
                '  Warning: Level 7 requested but no LLM model provided. '
                'Falling back to Level 6 behavior.'
            )

        # Create agents -- deliberately imbalanced roles to create tension
        # 2 contributors, 2 innovators, 1 curator, 1 maintainer
        # -> bug_fix and feature are over-represented, docs and review
        # under-served
        role_names = [
            'contributor', 'innovator', 'contributor', 'innovator',
            'knowledge_curator', 'maintainer',
        ]
        agent_names = ['Priya', 'Anya', 'Marcus', 'Jordan', 'Elena', 'Raj']
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

            # Level 7 with LLM: generate informed D-matrix priors from
            # backstory
            if self._use_llm:
                profile = social_data.AGENT_PROFILES.get(agent_names[i])
                if profile is not None:
                    backstory = profile.get('backstory', '')
                    role_label = profile.get(
                        'role', social_data.Role.CONTRIBUTOR
                    ).value
                    llm_d1 = aif.llm_generate_health_prior(
                        self.llm_model,
                        agent_names[i],
                        role_label,
                        backstory,
                    )
                    agent.D[0] = llm_d1
                    agent.beliefs[0] = llm_d1.copy()
                    print(
                        f'    LLM prior for {agent_names[i]}: '
                        f'healthy={llm_d1[0]:.2f} '
                        f'stressed={llm_d1[1]:.2f} '
                        f'declining={llm_d1[2]:.2f}'
                    )

            self.agents.append(agent)

        # Environment state (ground truth, hidden from agents)
        self.true_health = 0  # 0=healthy, 1=stressed, 2=declining
        self.true_urgency = 0  # 0=balanced, 1=bugs, 2=docs, 3=review

        # Stress schedule: stress hits at sprint 2 and 4 (only if enabled)
        self.stress_sprints: set[int] = set()
        if enable_stress:
            if num_sprints >= 3:
                self.stress_sprints.add(2)
            if num_sprints >= 5:
                self.stress_sprints.add(4)

        # Available tasks per sprint (not all types always available)
        self.available_tasks: list[str] = list(aif.ACTIONS[:4])

        # Metrics tracking
        self.history: list[dict[str, Any]] = []

    def _reward_params(
        self, agent: aif.ActiveInferenceAgent, action: str
    ) -> tuple[float, float, float]:
        """Return (success_prob, reward_if_success, reward_if_failure)."""
        if action == 'skip':
            return (1.0, 0.0, 0.0)

        if action not in self.available_tasks:
            return (1.0, -0.5, -0.5)  # deterministic penalty

        preferred = {
            'contributor': 'bug_fix',
            'innovator': 'feature',
            'knowledge_curator': 'documentation',
            'maintainer': 'code_review',
        }
        is_preferred = action == preferred.get(agent.role, '')

        # Success probability: 70% base, +15% if preferred, -20% if stressed
        success_prob = 0.70
        if is_preferred:
            success_prob += 0.15
        if self.true_health >= 2:  # declining
            success_prob -= 0.20
        elif self.true_health >= 1:  # stressed
            success_prob -= 0.10
        success_prob = float(np.clip(success_prob, 0.1, 0.95))

        # Project need: actions matching urgency get bonus
        urgency_map = {1: 'bug_fix', 2: 'documentation', 3: 'code_review'}
        matches_need = action == urgency_map.get(self.true_urgency, '')

        if is_preferred:
            reward_if_success = 3.0
        elif matches_need:
            reward_if_success = 2.5  # Urgency bonus
        else:
            reward_if_success = 1.0

        return (success_prob, reward_if_success, -1.0)

    def _get_expected_reward(
        self, agent: aif.ActiveInferenceAgent, action: str
    ) -> float:
        """Expected reward for action selection (no dice roll)."""
        p, r_success, r_failure = self._reward_params(agent, action)
        return p * r_success + (1 - p) * r_failure

    def _get_reward(
        self, agent: aif.ActiveInferenceAgent, action: str
    ) -> float:
        """Stochastic reward for actual execution (rolls the dice)."""
        p, r_success, r_failure = self._reward_params(agent, action)
        if self.rng.random() < p:
            return r_success
        return r_failure

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
        predicted_obs = np.clip(
            predicted_obs / predicted_obs.sum(), 1e-16, None
        )

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
                entropy -= state_prob * np.sum(
                    obs_probs * np.log(obs_probs)
                )

        return float(entropy)

    def _update_precision(
        self,
        agent: aif.ActiveInferenceAgent,
        prediction_error: float,
        sprint: int,
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
        action_counts: dict[str, int] = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        # Health requires active maintenance
        maintenance = (
            action_counts.get('bug_fix', 0)
            + action_counts.get('code_review', 0)
        )
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
        a_matrix = aif.build_A_matrix()
        probs = a_matrix[0][:, self.true_health, self.true_urgency]
        probs = probs / probs.sum()
        obs_idx = self.rng.choice(aif.NUM_HI_OBS, p=probs)
        return aif.HI_OBSERVATIONS[obs_idx]

    def run_sprint(self, sprint_num: int) -> dict[str, Any]:
        """Run one sprint through the experiment level's pipeline."""
        # 0. Apply stress events before anything else
        self._apply_stress_event(sprint_num)

        sprint_data: dict[str, Any] = {
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
        prediction_errors: dict[str, float] = {}
        for agent in self.agents:
            pe = self._compute_prediction_error(agent, hi_obs)
            prediction_errors[agent.name] = pe
        sprint_data['prediction_errors'] = prediction_errors

        # 4. Action selection
        actions: dict[str, str] = {}
        action_probs: dict[str, dict[str, float]] = {}
        for agent in self.agents:
            if self.level.use_efe_policy:
                # Level 4+: EFE-based policy selection
                action, probs = agent.decide()
            elif self.level.use_epistemic_value:
                # Level 2-3: Expected reward + epistemic bonus
                best_score = -float('inf')
                best_action = 'skip'
                probs = np.zeros(aif.NUM_ACTIONS)
                for a_idx, a_name in enumerate(aif.ACTIONS):
                    reward_est = self._get_expected_reward(agent, a_name)
                    epist = self._get_epistemic_bonus(agent, a_idx)
                    score = reward_est + 0.5 * epist
                    probs[a_idx] = score
                    if score > best_score:
                        best_score = score
                        best_action = a_name
                action = best_action
                probs = aif._softmax(probs)
            else:
                # Level 0-1: Pure expected-reward-based (greedy)
                best_reward = -float('inf')
                best_action = 'skip'
                probs = np.zeros(aif.NUM_ACTIONS)
                for a_name in aif.ACTIONS:
                    r = self._get_expected_reward(agent, a_name)
                    probs[aif.ACTIONS.index(a_name)] = r
                    if r > best_reward:
                        best_reward = r
                        best_action = a_name
                action = best_action
                probs = aif._softmax(probs)

            actions[agent.name] = action
            action_probs[agent.name] = {
                aif.ACTIONS[i]: float(probs[i])
                for i in range(aif.NUM_ACTIONS)
            }

        sprint_data['actions'] = actions
        sprint_data['action_probabilities'] = action_probs

        # 5. Environment dynamics
        self._simulate_environment_dynamics(list(actions.values()))

        # 6. Reward computation
        rewards: dict[str, float] = {}
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

        # 9. LLM-augmented observation processing (Level 7)
        #    After rewards are computed, ask the LLM to interpret the sprint
        #    narrative and blend its assessment with the hardcoded observation.
        #    This feeds back into belief updating for the *next* sprint.
        if self._use_llm:
            hi_bias, task_bias = aif.llm_interpret_sprint(
                self.llm_model,
                sprint_num,
                actions,
                rewards,
                hi_obs,
                aif.PROJECT_HEALTH_STATES[self.true_health],
            )
            # Blend the LLM observation with the hardcoded one and re-update
            # beliefs so the next sprint starts from an LLM-informed posterior.
            hi_obs_idx = aif.HI_OBSERVATIONS.index(hi_obs)
            task_obs_idx = aif.TASK_OBSERVATIONS.index(task_obs)
            blended_hi = aif.blend_observations(
                hi_obs_idx, hi_bias, aif.NUM_HI_OBS, mixing_weight=0.3,
            )
            blended_task = aif.blend_observations(
                task_obs_idx, task_bias, aif.NUM_TASK_OBS, mixing_weight=0.3,
            )
            blended_hi_name = aif.HI_OBSERVATIONS[blended_hi]
            blended_task_name = aif.TASK_OBSERVATIONS[blended_task]
            sprint_data['llm_blended_hi'] = blended_hi_name
            sprint_data['llm_blended_task'] = blended_task_name

            # Re-update beliefs with the blended observation
            for agent in self.agents:
                agent.observe(blended_hi_name, blended_task_name)

        # 10. Compute HI-like metric
        total_reward = sum(rewards.values())
        max_reward = len(self.agents) * 3.0
        hi = total_reward / max_reward if max_reward > 0 else 0.0

        # Diversity: how many different action types were chosen?
        unique_actions = len(
            set(a for a in actions.values() if a != 'skip')
        )
        diversity = unique_actions / len(social_data.TASK_TYPES)

        # Coverage: were all task types addressed?
        task_types_covered = set()
        for a in actions.values():
            if a in social_data.TASK_TYPES[:4]:
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
            'mean_coverage': float(np.mean(coverage_values)),
            'mean_diversity': float(np.mean(diversity_values)),
            'strategy_diversity': strategy_diversity,
            'hi_trajectory': hi_values,
            'sprint_history': self.history,
        }


# =============================================================================
# LLM Model Construction (for Level 7)
# =============================================================================


def _build_llm_model() -> Any:
    """Build an LLM model from CLI flags, or return None.

    Returns:
        A language model instance, or None if Level 7 LLM is not configured.
    """
    if FLAGS.use_mock:
        from concordia.testing import mock_model
        return mock_model.MockModel()

    if FLAGS.vllm_url:
        from concordia.contrib.language_models import vllm_remote
        return vllm_remote.VLLMModel(
            model_name=FLAGS.model_name,
            api_base=FLAGS.vllm_url,
        )

    return None


# =============================================================================
# Main
# =============================================================================


def print_comparison_table(results: list[dict]) -> None:
    """Print a comparison table across experiment levels."""
    print('\n' + '=' * 100)
    print('EXPERIMENT LADDER: RL -> ACTIVE INFERENCE')
    print('=' * 100)
    header = (
        f"{'Lvl':<4} {'Name':<30} {'Mean HI':<10} {'Coverage':<10} "
        f"{'Diversity':<10} {'Strat D':<10} {'New Concept':<30}"
    )
    print(header)
    print('-' * 100)

    for r in results:
        print(
            f"{r['level']:<4} {r['level_name']:<30} "
            f"{r['mean_hi']:<10.4f} "
            f"{r['mean_coverage']:<10.4f} "
            f"{r['mean_diversity']:<10.4f} "
            f"{r['strategy_diversity']:<10.4f} "
            f"{r['new_concept']:<30}"
        )

    print('=' * 100)

    # Show the conceptual mapping
    print('\nConceptual Bridge: RL -> Active Inference')
    print('-' * 80)
    for r in results:
        print(f"\nLevel {r['level']}: {r['level_name']}")
        print(f"  RL analog:       {r['rl_analog']}")
        print(f"  AIF mechanism:   {r['aif_mechanism']}")
        hi_traj = ' -> '.join(f'{h:.3f}' for h in r['hi_trajectory'])
        print(f'  HI trajectory:   {hi_traj}')


def ladder_results_to_evaluate_schema(ladder_result: dict) -> dict:
    """Convert experiment ladder output to evaluate.py schema.

    Mapping:
      mean_hi          -> harmony_index
      hi_trajectory    -> resilience_quotient (computed from stress recovery)
      rewards per sprint -> scores (cumulative per agent)
      actions          -> joint_action (rename in sprint_history)
      agent roles      -> player_roles
      stress sprints   -> dropout_name
    """
    # harmony_index: top-level from mean_hi
    hi = ladder_result.get('mean_hi', 0.0)

    # resilience_quotient: measure recovery from HI drops
    hi_traj = ladder_result.get('hi_trajectory', [])
    rq = 0.0
    if len(hi_traj) >= 3:
        max_drop = 0.0
        recovery_after_max_drop = 0.0
        for i in range(1, len(hi_traj)):
            drop = hi_traj[i - 1] - hi_traj[i]
            if drop > max_drop:
                max_drop = drop
                remaining_max = max(hi_traj[i:])
                recovery_after_max_drop = remaining_max - hi_traj[i]
        if max_drop > 0:
            rq = min(recovery_after_max_drop / max_drop, 1.0)

    # scores: cumulative reward per agent across sprints
    sprint_history = ladder_result.get('sprint_history', [])
    scores: dict[str, float] = {}
    for sprint in sprint_history:
        for agent_name, reward in sprint.get('rewards', {}).items():
            scores[agent_name] = scores.get(agent_name, 0.0) + reward

    # Rename 'actions' -> 'joint_action' in each sprint record
    adapted_history = []
    for sprint in sprint_history:
        entry = dict(sprint)
        if 'actions' in entry:
            entry['joint_action'] = entry.pop('actions')
        adapted_history.append(entry)

    # player_roles
    player_roles: dict[str, str] = {}
    role_names = [
        'contributor', 'innovator', 'contributor', 'innovator',
        'knowledge_curator', 'maintainer',
    ]
    agent_names = ['Priya', 'Anya', 'Marcus', 'Jordan', 'Elena', 'Raj']
    for name, role in zip(agent_names, role_names):
        player_roles[name] = role

    # dropout_name: set if any stress sprint occurred
    has_stress = any(
        s.get('is_stress_sprint', False) for s in sprint_history
    )
    dropout_name = 'stress_event' if has_stress else None

    return {
        'harmony_index': hi,
        'resilience_quotient': rq,
        'scores': scores,
        'sprint_history': adapted_history,
        'player_roles': player_roles,
        'dropout_name': dropout_name,
    }


def apply_ostrom_to_runner(
    runner: 'ExperimentRunner',
    principles: list[str] | None = None,
) -> None:
    """Apply Ostrom priors to all agents in an experiment runner."""
    for agent in runner.agents:
        aif.apply_ostrom_priors(agent, principles)


def run_ostrom_comparison(
    num_sprints: int,
    seed: int,
    principles: list[str] | None = None,
    enable_stress: bool = False,
) -> tuple[dict, dict]:
    """Run Level 4 (EFE) with and without Ostrom priors.

    Returns (baseline_result, ostrom_result).
    """
    level = EXPERIMENT_LEVELS[4]  # Level 4: Expected Free Energy

    # Baseline
    runner_base = ExperimentRunner(
        level=level,
        num_agents=6,
        num_sprints=num_sprints,
        seed=seed,
        enable_stress=enable_stress,
    )
    result_base = runner_base.run()

    # With Ostrom priors
    runner_ostrom = ExperimentRunner(
        level=level,
        num_agents=6,
        num_sprints=num_sprints,
        seed=seed,
        enable_stress=enable_stress,
    )
    apply_ostrom_to_runner(runner_ostrom, principles)
    result_ostrom = runner_ostrom.run()

    return result_base, result_ostrom


def print_ostrom_comparison(
    result_base: dict, result_ostrom: dict,
) -> None:
    """Print a comparison table for Ostrom priors."""
    print(f"\n{'=' * 60}")
    print('OSTROM PRIORS COMPARISON (Level 4: EFE)')
    print(f"{'=' * 60}")
    metrics = [
        ('Mean HI', 'mean_hi'),
        ('Coverage', 'mean_coverage'),
        ('Diversity', 'mean_diversity'),
        ('Strategy Div', 'strategy_diversity'),
    ]
    print(f"  {'Metric':<18} {'Baseline':>10} {'Ostrom':>10} {'Delta':>10}")
    print(f"  {'-' * 48}")
    for label, key in metrics:
        base_val = result_base.get(key, 0.0)
        ostrom_val = result_ostrom.get(key, 0.0)
        delta = ostrom_val - base_val
        pct = (delta / base_val * 100) if base_val != 0 else 0.0
        print(f"  {label:<18} {base_val:>10.4f} {ostrom_val:>10.4f} {delta:>+10.4f} ({pct:+.1f}%)")


def main(argv):
    del argv

    # Handle Ostrom comparison mode
    if FLAGS.ostrom:
        principles = None
        if FLAGS.ostrom_principles:
            principles = [p.strip() for p in FLAGS.ostrom_principles.split(',')]
        result_base, result_ostrom = run_ostrom_comparison(
            num_sprints=FLAGS.num_sprints,
            seed=FLAGS.seed,
            principles=principles,
            enable_stress=FLAGS.enable_stress,
        )
        print_ostrom_comparison(result_base, result_ostrom)
        # Save results
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        ostrom_file = os.path.join(FLAGS.output_dir, 'ostrom_comparison.json')
        with open(ostrom_file, 'w') as f:
            json.dump({
                'baseline': result_base,
                'ostrom': result_ostrom,
            }, f, indent=2, default=str)
        print(f'\nOstrom comparison saved to {ostrom_file}')
        return

    # Parse level specification
    if FLAGS.level.lower() == 'all':
        levels_to_run = list(range(len(EXPERIMENT_LEVELS)))
    else:
        levels_to_run = [int(x.strip()) for x in FLAGS.level.split(',')]

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Build LLM model if needed for Level 7
    needs_llm = any(
        EXPERIMENT_LEVELS[idx].use_llm_nodes
        for idx in levels_to_run
        if idx < len(EXPERIMENT_LEVELS)
    )
    llm_model = None
    if needs_llm:
        llm_model = _build_llm_model()
        if llm_model is None:
            print(
                '\nNote: Level 7 is in the run list but no LLM backend is '
                'configured. Use --use_mock for testing or --vllm_url for a '
                'real model. Level 7 will fall back to Level 6 behavior.\n'
            )

    results = []
    for level_idx in levels_to_run:
        if level_idx >= len(EXPERIMENT_LEVELS):
            print(f'Warning: Level {level_idx} not defined, skipping.')
            continue

        level = EXPERIMENT_LEVELS[level_idx]
        print(f"\n{'=' * 60}")
        print(f'Level {level.level}: {level.name}')
        print(f'New concept: {level.new_concept}')
        print(f"{'=' * 60}")

        runner = ExperimentRunner(
            level=level,
            num_agents=6,
            num_sprints=FLAGS.num_sprints,
            seed=FLAGS.seed,
            llm_model=llm_model if level.use_llm_nodes else None,
        )
        result = runner.run()
        results.append(result)

        print(f"  Mean HI:     {result['mean_hi']:.4f}")
        print(f"  Coverage:    {result['mean_coverage']:.4f}")
        print(f"  Diversity:   {result['mean_diversity']:.4f}")
        print(f"  Strategy D:  {result['strategy_diversity']:.4f}")
        print(f"  Duration:    {result['duration_s']:.2f}s")

        # Save individual result
        result_file = os.path.join(
            FLAGS.output_dir, f'level_{level_idx}.json'
        )
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    # Print comparison table
    if len(results) > 1:
        print_comparison_table(results)

    # Compute SustainScore if requested
    if FLAGS.compute_sustain_score:
        # evaluate.py defines duplicate absl flags. Mark all current flags
        # as overridable so the import doesn't crash.
        for _flag_name in list(FLAGS):
            FLAGS[_flag_name].allow_override = True
        from examples.games.sustain_hub import evaluate as _eval_mod
        compute_sustain_score = _eval_mod.compute_sustain_score
        print(f"\n{'=' * 60}")
        print('SUSTAIN SCORES')
        print(f"{'=' * 60}")
        for result in results:
            adapted = ladder_results_to_evaluate_schema(result)
            ss = compute_sustain_score(adapted)
            print(f"  Level {result['level']} ({result['level_name']}):")
            print(f"    SustainScore:   {ss['sustain_score']:.4f}")
            print(f"    HI:             {ss['harmony_index']:.4f}")
            print(f"    RQ:             {ss['resilience_quotient']:.4f}")
            print(f"    Fairness:       {ss['fairness']:.4f}")
            print(f"    Strategy Div:   {ss['strategy_diversity']:.4f}")

    # Save combined results
    combined_file = os.path.join(FLAGS.output_dir, 'ladder_results.json')
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {FLAGS.output_dir}/')


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
