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

"""Active Inference decision-making for SustainHub agents.

Implements the Free Energy Principle for agent decision-making, bridging
from the original SARSA RL approach (Rohira, 2025) to a full active
inference framework (Friston, Smith et al.).

The progression:
  1. LLM-only (current): Agent decides via natural language reasoning
  2. RL-augmented: LLM + Q-value memory influencing decisions
  3. Active Inference: LLM + Expected Free Energy minimization

Key reference:
  Smith, R., Friston, K.J., & Whyte, C.J. (2022). A Step-by-Step Tutorial
  on Active Inference and its Application to Empirical Data. Journal of
  Mathematical Psychology.
  https://github.com/rssmith33/Active-Inference-Tutorial-Scripts

POMDP structure for SustainHub:
  - Hidden states: project_health × task_urgency (2 factors)
  - Observations: harmony_index_level × task_completion_signal (2 modalities)
  - Actions/Policies: task_choice (bug_fix, feature, documentation, code_review, mentor, skip)
  - Time horizon: T=2 per sprint (observe → act)
"""

import math
import numpy as np
from typing import Any, Mapping, Sequence


# =============================================================================
# POMDP State/Observation Spaces
# =============================================================================

# Hidden state factor 1: Project Health (not directly observable)
PROJECT_HEALTH_STATES = ['healthy', 'stressed', 'declining']

# Hidden state factor 2: Task Urgency Profile
TASK_URGENCY_STATES = ['balanced', 'bugs_critical', 'docs_neglected', 'review_backlog']

# Observation modality 1: Harmony Index signal (what agent perceives)
HI_OBSERVATIONS = ['high', 'medium', 'low']

# Observation modality 2: Task completion feedback
TASK_OBSERVATIONS = ['success', 'partial', 'failure']

# Actions (policies the agent can select)
ACTIONS = ['bug_fix', 'feature', 'documentation', 'code_review', 'mentor', 'skip']

NUM_HEALTH = len(PROJECT_HEALTH_STATES)
NUM_URGENCY = len(TASK_URGENCY_STATES)
NUM_HI_OBS = len(HI_OBSERVATIONS)
NUM_TASK_OBS = len(TASK_OBSERVATIONS)
NUM_ACTIONS = len(ACTIONS)


# =============================================================================
# Generative Model Matrices
# =============================================================================

def build_A_matrix() -> list[np.ndarray]:
    """Build likelihood matrices: P(observation | hidden_states).

    A{1}: HI observation given (health, urgency) — shape (3, 3, 4)
    A{2}: Task outcome given (health, urgency) — shape (3, 3, 4)

    From Smith et al. tutorial: "Each column is a probability distribution
    that must sum to 1."
    """
    # A{1}: Harmony Index observation
    # When healthy + balanced → likely observe high HI
    # When declining + bugs_critical → likely observe low HI
    A1 = np.zeros((NUM_HI_OBS, NUM_HEALTH, NUM_URGENCY))

    for u in range(NUM_URGENCY):
        # Healthy state: mostly high HI, some medium
        A1[:, 0, u] = [0.7, 0.25, 0.05]
        # Stressed state: mostly medium HI
        A1[:, 1, u] = [0.15, 0.6, 0.25]
        # Declining state: mostly low HI
        A1[:, 2, u] = [0.05, 0.25, 0.7]

    # Urgency modulates: when urgency is critical, observations shift down
    # bugs_critical makes things look worse
    A1[:, :, 1] = np.roll(A1[:, :, 1], 1, axis=0)  # shift toward lower
    A1[:, :, 1] = A1[:, :, 1] / A1[:, :, 1].sum(axis=0, keepdims=True)

    # A{2}: Task outcome observation
    A2 = np.zeros((NUM_TASK_OBS, NUM_HEALTH, NUM_URGENCY))
    for u in range(NUM_URGENCY):
        A2[:, 0, u] = [0.7, 0.25, 0.05]   # healthy → likely success
        A2[:, 1, u] = [0.3, 0.5, 0.2]     # stressed → mixed
        A2[:, 2, u] = [0.1, 0.3, 0.6]     # declining → likely failure

    return [A1, A2]


def build_B_matrix() -> list[np.ndarray]:
    """Build transition matrices: P(next_state | current_state, action).

    B{1}: Health transitions — shape (3, 3, 6) per action
    B{2}: Urgency transitions — shape (4, 4, 6) per action

    Key insight: different actions affect state transitions differently.
    Bug fixes improve health when bugs are critical; features don't help
    when the project is declining.
    """
    # B{1}: Health state transitions per action
    B1 = np.zeros((NUM_HEALTH, NUM_HEALTH, NUM_ACTIONS))

    for a in range(NUM_ACTIONS):
        # Default: slight drift toward declining (entropy)
        B1[:, :, a] = np.array([
            [0.7, 0.2, 0.1],   # healthy → stays healthy, might stress
            [0.15, 0.6, 0.15], # stressed → might recover or decline
            [0.05, 0.2, 0.75], # declining → tends to stay declining
        ])

    # Bug fixes improve health (especially when stressed/declining)
    B1[:, :, ACTIONS.index('bug_fix')] = np.array([
        [0.8, 0.3, 0.1],
        [0.15, 0.5, 0.3],
        [0.05, 0.2, 0.6],
    ])

    # Code review improves health moderately
    B1[:, :, ACTIONS.index('code_review')] = np.array([
        [0.75, 0.25, 0.1],
        [0.2, 0.55, 0.25],
        [0.05, 0.2, 0.65],
    ])

    # Features: neutral to slightly negative for health
    B1[:, :, ACTIONS.index('feature')] = np.array([
        [0.65, 0.15, 0.05],
        [0.25, 0.55, 0.15],
        [0.1, 0.3, 0.8],
    ])

    # Mentoring: long-term health improvement
    B1[:, :, ACTIONS.index('mentor')] = np.array([
        [0.8, 0.25, 0.15],
        [0.15, 0.55, 0.3],
        [0.05, 0.2, 0.55],
    ])

    # Skip: project drifts toward decline
    B1[:, :, ACTIONS.index('skip')] = np.array([
        [0.5, 0.1, 0.05],
        [0.3, 0.5, 0.15],
        [0.2, 0.4, 0.8],
    ])

    # Normalize columns
    for a in range(NUM_ACTIONS):
        B1[:, :, a] = B1[:, :, a] / B1[:, :, a].sum(axis=0, keepdims=True)

    # B{2}: Urgency state transitions (simplified: less action-dependent)
    B2 = np.zeros((NUM_URGENCY, NUM_URGENCY, NUM_ACTIONS))
    for a in range(NUM_ACTIONS):
        # Default transition: urgency shifts around
        B2[:, :, a] = np.array([
            [0.5, 0.3, 0.3, 0.3],
            [0.2, 0.4, 0.1, 0.1],
            [0.2, 0.1, 0.4, 0.1],
            [0.1, 0.2, 0.2, 0.5],
        ])

    # Specific actions address specific urgencies
    # Bug fix resolves bugs_critical
    B2[0, 1, ACTIONS.index('bug_fix')] = 0.5  # bugs_critical → balanced
    B2[1, 1, ACTIONS.index('bug_fix')] = 0.2

    # Documentation resolves docs_neglected
    B2[0, 2, ACTIONS.index('documentation')] = 0.5
    B2[2, 2, ACTIONS.index('documentation')] = 0.2

    # Code review resolves review_backlog
    B2[0, 3, ACTIONS.index('code_review')] = 0.5
    B2[3, 3, ACTIONS.index('code_review')] = 0.2

    # Normalize
    for a in range(NUM_ACTIONS):
        B2[:, :, a] = B2[:, :, a] / B2[:, :, a].sum(axis=0, keepdims=True)

    return [B1, B2]


def build_C_matrix(role_preference: str = 'bug_fix') -> list[np.ndarray]:
    """Build preference matrices: log P(preferred observations).

    C{1}: Preference over HI observations (all agents prefer high HI)
    C{2}: Preference over task outcomes (all agents prefer success)

    From Smith et al.: "C vectors encode prior preferences over outcomes.
    Higher values indicate more preferred observations."

    The role_preference parameter biases task-outcome preferences toward
    the agent's preferred task type. Agents whose preferred task is a
    maintenance action (bug_fix, code_review) have a stronger aversion to
    failure (they care more about project health), while agents preferring
    growth actions (feature, documentation) have a stronger preference for
    success (they value visible output).
    """
    # All agents prefer high HI
    C1 = np.array([2.0, 0.0, -2.0])  # high=preferred, low=aversive

    # Base task outcome preferences
    C2 = np.array([2.0, 0.0, -2.0])  # success=preferred, failure=aversive

    # Role-specific bias on task outcome preferences
    maintenance_roles = {'bug_fix', 'code_review'}
    growth_roles = {'feature', 'documentation'}

    if role_preference in maintenance_roles:
        # Maintenance-oriented: stronger aversion to failure (project risk)
        C2[2] -= 1.0  # failure more aversive: -2 → -3
        C1[2] -= 0.5  # also more sensitive to low HI
    elif role_preference in growth_roles:
        # Growth-oriented: stronger preference for success (visible output)
        C2[0] += 1.0  # success more attractive: 2 → 3
        C1[0] += 0.5  # also value high HI more (visible progress)

    return [C1, C2]


def build_D_matrix(
    health_prior: str = 'uncertain',
    urgency_prior: str = 'uncertain',
) -> list[np.ndarray]:
    """Build prior state beliefs: P(initial hidden states).

    D{1}: Prior beliefs about project health
    D{2}: Prior beliefs about task urgency

    These are Dirichlet concentration parameters — higher values = more
    confident. They update after each sprint based on posterior beliefs.
    """
    if health_prior == 'uncertain':
        D1 = np.array([1.0, 1.0, 1.0])  # uniform/uncertain
    elif health_prior == 'optimistic':
        D1 = np.array([4.0, 1.0, 0.5])  # believes project is healthy
    elif health_prior == 'pessimistic':
        D1 = np.array([0.5, 1.0, 4.0])  # believes project is declining
    else:
        D1 = np.ones(NUM_HEALTH)

    if urgency_prior == 'uncertain':
        D2 = np.ones(NUM_URGENCY)
    else:
        D2 = np.ones(NUM_URGENCY)

    # Normalize to probability distributions
    D1 = D1 / D1.sum()
    D2 = D2 / D2.sum()

    return [D1, D2]


def build_E_matrix(
    role: str = 'contributor',
    habit_strength: float = 1.0,
) -> np.ndarray:
    """Build policy prior (habits): E-vector.

    Encodes learned policy preferences from past experience.
    Role-specific habits bias toward preferred task types.

    This is where the RL → Active Inference bridge happens:
    Q-values from SARSA map onto log(E) as habitual policy priors.
    """
    # Start with uniform policy prior
    E = np.ones(NUM_ACTIONS)

    # Role-specific habit biases
    role_habits = {
        'contributor': {'bug_fix': 3.0, 'feature': 1.0},
        'innovator': {'feature': 3.0, 'bug_fix': 1.0},
        'knowledge_curator': {'documentation': 3.0, 'code_review': 1.0},
        'maintainer': {'code_review': 3.0, 'documentation': 1.0},
    }

    if role.lower() in role_habits:
        for action, bonus in role_habits[role.lower()].items():
            idx = ACTIONS.index(action)
            E[idx] += bonus * habit_strength

    E = E / E.sum()
    return E


# =============================================================================
# Active Inference Core: Expected Free Energy
# =============================================================================

def compute_expected_free_energy(
    A: list[np.ndarray],
    B: list[np.ndarray],
    C: list[np.ndarray],
    D: list[np.ndarray],
    action_idx: int,
    current_beliefs: list[np.ndarray],
    gamma: float = 1.0,
) -> float:
    """Compute Expected Free Energy (EFE) for a given action.

    G(π) = E_Q[ln Q(s') - ln P(o', s')]
         = -E_Q[ln P(o'|s')] - E_Q[H[P(o'|s')]]
         = -pragmatic_value   - epistemic_value

    Where:
      pragmatic_value = expected reward (how well outcomes match preferences)
      epistemic_value = expected information gain (how much uncertainty reduces)

    This is the key equation from Smith et al. (2022), Eq. 2.9.

    Args:
        A: Likelihood matrices
        B: Transition matrices
        C: Preference vectors (log preferences)
        D: Prior beliefs
        action_idx: Index of the action to evaluate
        current_beliefs: Current posterior beliefs about hidden states
        gamma: Precision parameter (inverse temperature). Higher = more
               exploitative, lower = more exploratory.

    Returns:
        G: Expected free energy (lower is better — agent minimizes this)
    """
    # Predict next state given action: Q(s') = B(a) @ Q(s)
    predicted_health = B[0][:, :, action_idx] @ current_beliefs[0]
    predicted_urgency = B[1][:, :, action_idx] @ current_beliefs[1]

    G = 0.0

    # For each observation modality
    for m, Am in enumerate(A):
        Cm = C[m]

        # Predicted observations: P(o'|s') averaged over predicted states
        # For simplicity, marginalize over urgency for HI observations
        predicted_obs = np.zeros(Am.shape[0])
        for h in range(NUM_HEALTH):
            for u in range(NUM_URGENCY):
                predicted_obs += Am[:, h, u] * predicted_health[h] * predicted_urgency[u]

        predicted_obs = np.clip(predicted_obs, 1e-16, None)
        predicted_obs = predicted_obs / predicted_obs.sum()

        # Pragmatic value: E_Q[ln P(o')] — do predicted observations match preferences?
        log_preferences = Cm - _log_sum_exp(Cm)  # normalize to log probabilities
        pragmatic = np.sum(predicted_obs * log_preferences)

        # Epistemic value: -E_Q[H[P(o'|s')]] — does this action reduce uncertainty?
        # H = -sum P(o|s) ln P(o|s) averaged over predicted states
        entropy = 0.0
        for h in range(NUM_HEALTH):
            for u in range(NUM_URGENCY):
                obs_given_state = np.clip(Am[:, h, u], 1e-16, None)
                state_prob = predicted_health[h] * predicted_urgency[u]
                entropy -= state_prob * np.sum(obs_given_state * np.log(obs_given_state))
        epistemic = entropy  # negative entropy = information gain

        G += -gamma * pragmatic - epistemic

    return G


def select_action(
    A: list[np.ndarray],
    B: list[np.ndarray],
    C: list[np.ndarray],
    D: list[np.ndarray],
    E: np.ndarray,
    current_beliefs: list[np.ndarray],
    gamma: float = 1.0,
    alpha: float = 16.0,
) -> tuple[int, np.ndarray]:
    """Select action by minimizing Expected Free Energy.

    P(π) = σ(-G(π) + ln E(π))

    Where σ is softmax with precision alpha.

    Args:
        A, B, C, D: Generative model matrices
        E: Policy prior (habits)
        current_beliefs: Current beliefs about hidden states
        gamma: EFE precision (exploitation vs exploration)
        alpha: Action precision (how deterministic the policy is)

    Returns:
        action_idx: Selected action index
        action_probs: Probability distribution over actions
    """
    G = np.zeros(NUM_ACTIONS)

    for a in range(NUM_ACTIONS):
        G[a] = compute_expected_free_energy(
            A, B, C, D, a, current_beliefs, gamma
        )

    # Combine EFE with habit prior
    log_policy = -G + np.log(np.clip(E, 1e-16, None))

    # Softmax with precision
    action_probs = _softmax(alpha * log_policy)

    # Sample action
    action_idx = np.random.choice(NUM_ACTIONS, p=action_probs)

    return action_idx, action_probs


def update_beliefs(
    A: list[np.ndarray],
    current_beliefs: list[np.ndarray],
    observations: list[int],
    num_iterations: int = 16,
) -> list[np.ndarray]:
    """Update beliefs about hidden states given new observations.

    Implements variational message passing (approximate Bayesian inference):
    Q(s) ∝ P(o|s) × P(s)

    This is the perception step — the agent infers the hidden state of the
    project from its observations.

    Args:
        A: Likelihood matrices
        current_beliefs: Prior beliefs (before observation)
        observations: List of observation indices [hi_obs, task_obs]
        num_iterations: Number of variational iterations

    Returns:
        Updated beliefs about hidden states
    """
    # Start from current beliefs
    beliefs = [b.copy() for b in current_beliefs]

    for _ in range(num_iterations):
        # Log beliefs
        log_beliefs = [np.log(np.clip(b, 1e-16, None)) for b in beliefs]

        # Update health beliefs using both observation modalities
        log_health = np.zeros(NUM_HEALTH)
        for m, (Am, obs) in enumerate(zip(A, observations)):
            # Marginalize over urgency
            for u in range(NUM_URGENCY):
                log_health += beliefs[1][u] * np.log(np.clip(Am[obs, :, u], 1e-16, None))
        log_health += log_beliefs[0]  # prior
        beliefs[0] = _softmax(log_health)

        # Update urgency beliefs
        log_urgency = np.zeros(NUM_URGENCY)
        for m, (Am, obs) in enumerate(zip(A, observations)):
            for h in range(NUM_HEALTH):
                log_urgency += beliefs[0][h] * np.log(np.clip(Am[obs, h, :], 1e-16, None))
        log_urgency += log_beliefs[1]
        beliefs[1] = _softmax(log_urgency)

    return beliefs


def update_habits(
    E: np.ndarray,
    action_idx: int,
    outcome_valence: float,
    learning_rate: float = 0.1,
) -> np.ndarray:
    """Update policy prior (habits) based on action outcome.

    This is the RL → Active Inference bridge:
    - In SARSA: Q(s,a) += α × (r + γQ(s',a') - Q(s,a))
    - Here: E(a) += η × outcome_valence

    Dirichlet concentration parameters accumulate evidence for good policies.

    Args:
        E: Current habit vector
        action_idx: Action that was taken
        outcome_valence: How good the outcome was (-1 to +1)
        learning_rate: How fast habits update (η)

    Returns:
        Updated habit vector
    """
    E_new = E.copy()
    # Increase concentration for rewarded actions
    E_new[action_idx] += learning_rate * max(0, outcome_valence)
    # Slight decrease for punished actions (but floor at small positive value)
    if outcome_valence < 0:
        E_new[action_idx] = max(0.01, E_new[action_idx] + learning_rate * outcome_valence)
    # Normalize
    E_new = E_new / E_new.sum()
    return E_new


# =============================================================================
# Agent-Level Integration
# =============================================================================

class ActiveInferenceAgent:
    """An agent that uses active inference for decision-making.

    This can be used alongside the LLM agent in Concordia:
    1. LLM generates natural language reasoning about the situation
    2. ActiveInferenceAgent computes action probabilities from the POMDP
    3. The two signals are combined (e.g., LLM proposes, AIF validates)

    The agent maintains:
    - Generative model (A, B, C, D, E matrices)
    - Current beliefs about hidden states
    - Habit memory (E-vector, updated across sprints)
    """

    def __init__(
        self,
        name: str,
        role: str,
        gamma: float = 1.0,
        alpha: float = 16.0,
        learning_rate: float = 0.1,
        health_prior: str = 'uncertain',
    ):
        self.name = name
        self.role = role
        self.gamma = gamma  # EFE precision
        self.alpha = alpha  # action precision
        self.learning_rate = learning_rate

        # Build generative model
        role_to_preferred_task = {
            'contributor': 'bug_fix',
            'innovator': 'feature',
            'knowledge_curator': 'documentation',
            'maintainer': 'code_review',
        }
        self.A = build_A_matrix()
        self.B = build_B_matrix()
        self.C = build_C_matrix(
            role_preference=role_to_preferred_task.get(role, 'bug_fix')
        )
        self.D = build_D_matrix(health_prior=health_prior)
        self.E = build_E_matrix(role=role)

        # Current beliefs (start from priors)
        self.beliefs = [d.copy() for d in self.D]

        # History for analysis
        self.action_history = []
        self.belief_history = []
        self.efe_history = []

    def observe(self, hi_level: str, task_outcome: str) -> None:
        """Update beliefs based on new observations."""
        hi_idx = HI_OBSERVATIONS.index(hi_level)
        task_idx = TASK_OBSERVATIONS.index(task_outcome)
        self.beliefs = update_beliefs(self.A, self.beliefs, [hi_idx, task_idx])
        self.belief_history.append([b.copy() for b in self.beliefs])

    def decide(self) -> tuple[str, np.ndarray]:
        """Select an action using expected free energy minimization."""
        action_idx, action_probs = select_action(
            self.A, self.B, self.C, self.D, self.E,
            self.beliefs, self.gamma, self.alpha,
        )
        action = ACTIONS[action_idx]
        self.action_history.append(action)
        self.efe_history.append(action_probs.copy())
        return action, action_probs

    def learn(self, outcome_valence: float) -> None:
        """Update habits based on sprint outcome."""
        if self.action_history:
            last_action_idx = ACTIONS.index(self.action_history[-1])
            self.E = update_habits(
                self.E, last_action_idx, outcome_valence, self.learning_rate
            )

    def get_state_summary(self) -> dict[str, Any]:
        """Return a summary of the agent's internal state for logging."""
        return {
            'name': self.name,
            'role': self.role,
            'beliefs_health': {
                PROJECT_HEALTH_STATES[i]: float(self.beliefs[0][i])
                for i in range(NUM_HEALTH)
            },
            'beliefs_urgency': {
                TASK_URGENCY_STATES[i]: float(self.beliefs[1][i])
                for i in range(NUM_URGENCY)
            },
            'habits': {
                ACTIONS[i]: float(self.E[i])
                for i in range(NUM_ACTIONS)
            },
            'gamma': self.gamma,
            'alpha': self.alpha,
        }


# =============================================================================
# LLM + Active Inference Hybrid
# =============================================================================

# =============================================================================
# LLM-as-Node: LLM integration points for the AIF factor graph (Level 7)
# =============================================================================

def llm_generate_health_prior(
    model: Any,
    agent_name: str,
    agent_role: str,
    agent_backstory: str,
) -> np.ndarray:
    """Use the LLM to generate an informed D-matrix prior from backstory.

    Instead of a fixed 'uncertain'/'optimistic'/'pessimistic' prior, the LLM
    reads the agent's backstory and personality to infer how they would initially
    assess the project's health. This makes each agent's prior genuinely
    individual.

    Args:
        model: A language model implementing sample_choice().
        agent_name: The agent's name.
        agent_role: The agent's role (e.g. 'contributor').
        agent_backstory: The agent's full backstory text.

    Returns:
        D1: A normalized probability distribution over health states
            [healthy, stressed, declining].
    """
    prompt = (
        f"You are assessing the mindset of {agent_name}, a {agent_role} in an "
        f"open-source project called SustainHub.\n\n"
        f"Backstory: {agent_backstory}\n\n"
        f"Based on this person's background and personality, what would be "
        f"their initial gut assessment of the project's health before seeing "
        f"any data? Consider: optimists who trust the community might lean "
        f"'healthy'; burned-out or skeptical people might lean 'stressed' or "
        f"'declining'; newcomers or uncertain people might be neutral.\n\n"
        f"Rate the project health as one of: healthy, stressed, declining"
    )

    try:
        idx, _, _ = model.sample_choice(
            prompt,
            ['healthy', 'stressed', 'declining'],
        )
    except Exception:
        # Fallback to uncertain prior on any LLM error
        return np.array([1.0, 1.0, 1.0]) / 3.0

    # Map LLM choice to a prior distribution (concentrated but not degenerate)
    prior_map = {
        0: np.array([4.0, 1.5, 0.5]),  # healthy
        1: np.array([1.0, 4.0, 1.0]),  # stressed
        2: np.array([0.5, 1.5, 4.0]),  # declining
    }
    D1 = prior_map[idx]
    D1 = D1 / D1.sum()
    return D1


def llm_interpret_sprint(
    model: Any,
    sprint_num: int,
    actions: dict[str, str],
    rewards: dict[str, float],
    hi_observation: str,
    true_health_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the LLM to interpret sprint results as observation biases.

    After each sprint, the LLM reads a narrative summary of what happened
    and produces soft estimates of project health and task completion quality.
    These are blended with the hardcoded observation to produce a richer signal.

    Args:
        model: A language model implementing sample_choice().
        sprint_num: Current sprint number.
        actions: Dict mapping agent_name -> action taken.
        rewards: Dict mapping agent_name -> reward received.
        hi_observation: The hardcoded HI observation string ('high'/'medium'/'low').
        true_health_label: Label of true health state (for narrative only).

    Returns:
        hi_bias: Probability distribution over HI observations [high, medium, low].
        task_bias: Probability distribution over task observations [success, partial, failure].
    """
    # Build a sprint narrative for the LLM
    total_reward = sum(rewards.values())
    num_agents = len(actions)
    action_summary = ', '.join(
        f"{name}: {act}" for name, act in actions.items()
    )
    reward_summary = ', '.join(
        f"{name}: {r:+.1f}" for name, r in rewards.items()
    )

    health_prompt = (
        f"Sprint {sprint_num} just finished in the SustainHub project.\n\n"
        f"Actions taken: {action_summary}\n"
        f"Scores received: {reward_summary}\n"
        f"Total team score: {total_reward:+.1f} (out of max {num_agents * 3.0:.0f})\n"
        f"Observed harmony index: {hi_observation}\n\n"
        f"Based on this sprint's results, how would you rate the project's "
        f"overall health? Consider: did the team cover diverse needs, or did "
        f"they cluster on preferred tasks? Were scores generally positive?\n\n"
        f"Rate the project health as: high, medium, or low"
    )

    task_prompt = (
        f"Sprint {sprint_num} just finished in the SustainHub project.\n\n"
        f"Actions taken: {action_summary}\n"
        f"Scores received: {reward_summary}\n\n"
        f"Based on the scores and action choices, how would you rate the "
        f"overall task completion quality this sprint?\n\n"
        f"Rate task completion as: success, partial, or failure"
    )

    # Query LLM for health assessment
    try:
        hi_idx, _, _ = model.sample_choice(
            health_prompt, ['high', 'medium', 'low']
        )
    except Exception:
        hi_idx = HI_OBSERVATIONS.index(hi_observation)  # fallback

    # Query LLM for task assessment
    try:
        task_idx, _, _ = model.sample_choice(
            task_prompt, ['success', 'partial', 'failure']
        )
    except Exception:
        task_idx = 0  # fallback to success

    # Convert to soft distributions (concentrated around chosen category)
    def _choice_to_dist(idx: int, n: int, concentration: float = 3.0) -> np.ndarray:
        dist = np.ones(n) * 0.5
        dist[idx] = concentration
        return dist / dist.sum()

    hi_bias = _choice_to_dist(hi_idx, NUM_HI_OBS)
    task_bias = _choice_to_dist(task_idx, NUM_TASK_OBS)

    return hi_bias, task_bias


def blend_observations(
    hardcoded_obs_idx: int,
    llm_bias: np.ndarray,
    num_categories: int,
    mixing_weight: float = 0.3,
) -> int:
    """Blend a hardcoded observation with an LLM-derived bias.

    Args:
        hardcoded_obs_idx: Index of the observation from the generative model.
        llm_bias: LLM-derived probability distribution over observations.
        num_categories: Number of observation categories.
        mixing_weight: How much to weight the LLM (0.0 = ignore LLM, 1.0 = only LLM).

    Returns:
        Blended observation index (sampled from the mixed distribution).
    """
    # One-hot for the hardcoded observation
    hardcoded_dist = np.zeros(num_categories)
    hardcoded_dist[hardcoded_obs_idx] = 1.0

    # Blend
    blended = (1.0 - mixing_weight) * hardcoded_dist + mixing_weight * llm_bias
    blended = blended / blended.sum()

    # Sample from blended distribution
    return int(np.random.choice(num_categories, p=blended))


def format_aif_context_for_llm(agent: ActiveInferenceAgent) -> str:
    """Format active inference state as context for LLM prompting.

    This allows the LLM agent to "see" what the active inference model
    thinks, creating a hybrid decision system where:
    - AIF provides Bayesian-rational policy probabilities
    - LLM provides natural language reasoning and social awareness
    """
    state = agent.get_state_summary()
    health_beliefs = state['beliefs_health']
    urgency_beliefs = state['beliefs_urgency']
    habits = state['habits']

    # Find most likely health state
    top_health = max(health_beliefs, key=health_beliefs.get)
    top_urgency = max(urgency_beliefs, key=urgency_beliefs.get)

    # Format action probabilities
    sorted_actions = sorted(habits.items(), key=lambda x: x[1], reverse=True)
    action_lines = [f"  {a}: {p:.0%}" for a, p in sorted_actions[:3]]

    return (
        f"[Internal Assessment]\n"
        f"Project health belief: {top_health} "
        f"({health_beliefs[top_health]:.0%} confidence)\n"
        f"Current urgency: {top_urgency} "
        f"({urgency_beliefs[top_urgency]:.0%} confidence)\n"
        f"Learned task preferences:\n" + "\n".join(action_lines)
    )


# =============================================================================
# Utility Functions
# =============================================================================

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _log_sum_exp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))
