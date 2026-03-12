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
flags.DEFINE_integer('num_sprints', 3, 'Number of sprints per experiment.')
flags.DEFINE_integer('seed', 42, 'Random seed.')


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
        self.rng = np.random.RandomState(seed)

        # Create agents — deliberately imbalanced roles to create tension
        # 2 contributors, 2 innovators, 1 curator, 1 maintainer
        # → bug_fix and feature are over-represented, docs and review under-served
        role_names = ['contributor', 'innovator', 'contributor', 'innovator',
                      'knowledge_curator', 'maintainer']
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
                action, probs = agent.decide()  # returns (action_string, probs)
            elif self.level.use_epistemic_value:
                # Level 2-3: Reward + epistemic bonus
                best_score = -float('inf')
                best_action = 'skip'
                probs = np.zeros(aif.NUM_ACTIONS)
                for a_idx, a_name in enumerate(aif.ACTIONS):
                    reward_est = self._get_reward(agent, a_name)
                    epist = self._get_epistemic_bonus(agent, a_idx)
                    score = reward_est + 0.5 * epist
                    probs[a_idx] = score
                    if score > best_score:
                        best_score = score
                        best_action = a_name
                action = best_action
                probs = aif._softmax(probs)
            else:
                # Level 0-1: Pure reward-based (greedy)
                best_reward = -float('inf')
                best_action = 'skip'
                probs = np.zeros(aif.NUM_ACTIONS)
                for a_name in aif.ACTIONS:
                    r = self._get_reward(agent, a_name)
                    probs[aif.ACTIONS.index(a_name)] = r
                    if r > best_reward:
                        best_reward = r
                        best_action = a_name
                action = best_action
                probs = aif._softmax(probs)

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
            num_agents=6,
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


if __name__ == '__main__':
    app.run(main)
