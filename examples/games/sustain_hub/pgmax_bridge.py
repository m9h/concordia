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

"""Active Inference bridge for SustainHub.

This module bridges SustainHub's domain-specific POMDP (active_inference.py)
to ALF (Active inference/Learning Framework) or PGMax's GPU-accelerated
Active Inference framework, enabling JAX-backed inference and policy selection.

Preferred backend: ALF (standalone JAX-native, analytic EFE)
Fallback 1: PGMax (BP-based inference)
Fallback 2: NumPy (pure numpy EFE computation)

STATE SPACE MAPPING
-------------------

SustainHub's hand-rolled AIF uses a 2-factor POMDP:

  Factor 1: project_health  -- 3 states: {healthy, stressed, declining}
  Factor 2: task_urgency    -- 4 states: {balanced, bugs_critical,
                                          docs_neglected, review_backlog}

PGMax Phase 1 supports a single discrete hidden-state factor. To bridge this
gap, we flatten the 2-factor state space into a single factor with
3 x 4 = 12 composite states:

  state_index = health_index * NUM_URGENCY + urgency_index

Observation modalities are similarly flattened from 2 modalities into 1:

  SustainHub has 2 modalities: HI_level (3 values) x task_outcome (3 values)
  Flattened: 3 x 3 = 9 composite observation indices.
    obs_index = hi_index * NUM_TASK_OBS + task_index

Actions are kept as-is (6 actions: bug_fix, feature, documentation,
code_review, mentor, skip).

MATRIX CONSTRUCTION
-------------------

  A matrix (likelihood): PGMax list with 1 element, shape (9, 12)
    A[0][o, s] = P(obs=o | state=s), built by multiplying the two original
    A-matrix modalities and flattening.

  B matrix (transitions): PGMax list with 1 element, shape (12, 12, 6)
    B[0][s', s, a] = P(s'|s, a), built as Kronecker product of B1 and B2.

  C vector (preferences): PGMax list with 1 element, shape (9,)
    C[0][o] = log-preference, built as outer sum of C1 and C2 then flattened.

  D vector (state prior): PGMax list with 1 element, shape (12,)
    D[0][s] = prior, built as outer product of D1 and D2 then flattened.

  E vector (policy prior): shape (6,) for T=1 single-factor (one policy per
    action). Built from SustainHub's role-specific habits.

UPGRADING TO MULTI-FACTOR
--------------------------

When PGMax adds multi-factor support, the bridge can be upgraded by:

  1. Passing A as a list of 2 matrices with original shapes:
       A[0]: (3, 3, 4)  -- HI obs given (health, urgency)
       A[1]: (3, 3, 4)  -- task obs given (health, urgency)
  2. Passing B as a list of 2 matrices:
       B[0]: (3, 3, 6)  -- health transitions per action
       B[1]: (4, 4, 6)  -- urgency transitions per action
  3. Passing C as a list of 2 vectors:
       C[0]: (3,)  -- HI preferences
       C[1]: (3,)  -- task outcome preferences
  4. Passing D as a list of 2 vectors:
       D[0]: (3,)  -- health prior
       D[1]: (4,)  -- urgency prior

  The encoding/decoding helpers and get_factored_beliefs will become
  unnecessary since PGMax will handle per-factor marginals natively.

HOW TO SWAP BETWEEN HAND-ROLLED AND PGMAX AIF
----------------------------------------------

In experiments.py, replace the agent construction:

  # Hand-rolled (current):
  from examples.games.sustain_hub import active_inference as aif
  agent = aif.ActiveInferenceAgent(name=..., role=..., ...)

  # PGMax-backed (new):
  from examples.games.sustain_hub import pgmax_bridge
  agent = pgmax_bridge.PGMaxAgent(name=..., role=..., ...)

PGMaxAgent has the same external interface as ActiveInferenceAgent:
  - agent.observe(hi_level: str, task_outcome: str)
  - agent.decide() -> (action_str, action_probs)
  - agent.learn(outcome_valence: float)
  - agent.get_state_summary() -> dict
"""

import sys
import os
import warnings
from typing import Any

import numpy as np

from examples.games.sustain_hub import active_inference as aif

# ---------------------------------------------------------------------------
# Backend imports: prefer ALF > PGMax > NumPy fallback
# ---------------------------------------------------------------------------

_ALF_AVAILABLE = False
_PGMAX_AVAILABLE = False
_backend = 'numpy'

# Try ALF first (standalone JAX-native Active Inference)
try:
    from alf.generative_model import GenerativeModel
    from alf.agent import AnalyticAgent as _ALFAgent
    from alf.sequential_efe import evaluate_all_policies_sequential
    from alf.policy import select_action as _alf_select_action
    _ALF_AVAILABLE = True
    _backend = 'alf'
except ImportError:
    _ALFAgent = None

# Try PGMax as fallback (BP-based Active Inference)
if not _ALF_AVAILABLE:
    try:
        from pgmax.aif.generative_model import GenerativeModel
        from pgmax.aif.agent import ActiveInferenceAgent as _PGMaxAIFAgent
        _PGMAX_AVAILABLE = True
        _backend = 'pgmax'
    except ImportError:
        _pgmax_dev_path = os.path.expanduser('/Users/mhough/dev/PGMax')
        if os.path.isdir(_pgmax_dev_path) and _pgmax_dev_path not in sys.path:
            sys.path.insert(0, _pgmax_dev_path)
        try:
            from pgmax.aif.generative_model import GenerativeModel
            from pgmax.aif.agent import ActiveInferenceAgent as _PGMaxAIFAgent
            _PGMAX_AVAILABLE = True
            _backend = 'pgmax'
        except ImportError:
            GenerativeModel = None
            _PGMaxAIFAgent = None
else:
    _PGMaxAIFAgent = None


def alf_available() -> bool:
    """Return True if ALF is importable."""
    return _ALF_AVAILABLE


def pgmax_available() -> bool:
    """Return True if PGMax AIF is importable."""
    return _PGMAX_AVAILABLE


def get_backend() -> str:
    """Return the active backend name: 'alf', 'pgmax', or 'numpy'."""
    return _backend


# ---------------------------------------------------------------------------
# Domain constants (re-exported from active_inference.py for convenience)
# ---------------------------------------------------------------------------

PROJECT_HEALTH_STATES = aif.PROJECT_HEALTH_STATES   # 3 states
TASK_URGENCY_STATES = aif.TASK_URGENCY_STATES       # 4 states
HI_OBSERVATIONS = aif.HI_OBSERVATIONS               # 3 levels
TASK_OBSERVATIONS = aif.TASK_OBSERVATIONS            # 3 levels
ACTIONS = aif.ACTIONS                                # 6 actions

NUM_HEALTH = aif.NUM_HEALTH
NUM_URGENCY = aif.NUM_URGENCY
NUM_HI_OBS = aif.NUM_HI_OBS
NUM_TASK_OBS = aif.NUM_TASK_OBS
NUM_ACTIONS = aif.NUM_ACTIONS

# Composite (flattened) dimensions
NUM_COMPOSITE_STATES = NUM_HEALTH * NUM_URGENCY       # 12
NUM_COMPOSITE_OBS = NUM_HI_OBS * NUM_TASK_OBS         # 9


# ===========================================================================
# Encoding / Decoding helpers
# ===========================================================================

def encode_state(health_idx: int, urgency_idx: int) -> int:
    """Encode a (health, urgency) pair into a single composite state index.

    Layout: row-major with urgency as the fast axis.
    """
    return health_idx * NUM_URGENCY + urgency_idx


def decode_state(composite_idx: int) -> tuple[int, int]:
    """Decode a composite state index into (health_idx, urgency_idx)."""
    return divmod(composite_idx, NUM_URGENCY)


def encode_observation(hi_idx: int, task_idx: int) -> int:
    """Encode (hi_obs, task_obs) into a single composite observation index."""
    return hi_idx * NUM_TASK_OBS + task_idx


def decode_observation(composite_idx: int) -> tuple[int, int]:
    """Decode a composite observation index into (hi_idx, task_idx)."""
    return divmod(composite_idx, NUM_TASK_OBS)


def action_str_to_idx(action: str) -> int:
    """Convert an action string to its integer index."""
    return ACTIONS.index(action)


def action_idx_to_str(idx: int) -> str:
    """Convert an integer action index to its string name."""
    return ACTIONS[idx]


def hi_str_to_idx(hi_level: str) -> int:
    """Convert a Harmony Index level string to its index."""
    return HI_OBSERVATIONS.index(hi_level)


def task_str_to_idx(task_outcome: str) -> int:
    """Convert a task outcome string to its index."""
    return TASK_OBSERVATIONS.index(task_outcome)


def discretize_hi(hi_value: float) -> str:
    """Discretize a continuous harmony index [0,1] into {high, medium, low}."""
    if hi_value >= 0.7:
        return 'high'
    elif hi_value >= 0.4:
        return 'medium'
    else:
        return 'low'


def discretize_task_outcome(reward: float) -> str:
    """Discretize a reward value into {success, partial, failure}."""
    if reward >= 2.0:
        return 'success'
    elif reward >= 0.0:
        return 'partial'
    else:
        return 'failure'


def get_factored_beliefs(composite_beliefs: np.ndarray) -> list[np.ndarray]:
    """Marginalize composite beliefs back into per-factor beliefs.

    Given a distribution over 12 composite states, returns:
      [health_beliefs(3,), urgency_beliefs(4,)]
    """
    beliefs_2d = composite_beliefs.reshape(NUM_HEALTH, NUM_URGENCY)
    health_beliefs = beliefs_2d.sum(axis=1)
    urgency_beliefs = beliefs_2d.sum(axis=0)

    # Normalize
    health_sum = health_beliefs.sum()
    urgency_sum = urgency_beliefs.sum()
    if health_sum > 0:
        health_beliefs = health_beliefs / health_sum
    if urgency_sum > 0:
        urgency_beliefs = urgency_beliefs / urgency_sum

    return [health_beliefs, urgency_beliefs]


def flatten_factor_beliefs(
    health_beliefs: np.ndarray,
    urgency_beliefs: np.ndarray,
) -> np.ndarray:
    """Build a composite belief vector from independent factor beliefs.

    Assumes mean-field factorization: P(h, u) = P(h) * P(u).

    Returns:
        Array of shape (NUM_COMPOSITE_STATES,), sums to 1.
    """
    joint = np.outer(health_beliefs, urgency_beliefs).flatten()
    return joint / joint.sum()


# ===========================================================================
# Flattened matrix construction
# ===========================================================================

def build_flat_A_matrix() -> np.ndarray:
    """Build the flattened likelihood matrix A: shape (9, 12).

    A[o, s] = P(composite_obs=o | composite_state=s)

    Constructed by multiplying the two modality likelihoods (assuming
    conditional independence given state) and flattening both axes.
    """
    A_factors = aif.build_A_matrix()  # [A1(3,3,4), A2(3,3,4)]
    A1 = A_factors[0]  # (NUM_HI_OBS, NUM_HEALTH, NUM_URGENCY)
    A2 = A_factors[1]  # (NUM_TASK_OBS, NUM_HEALTH, NUM_URGENCY)

    A_flat = np.zeros((NUM_COMPOSITE_OBS, NUM_COMPOSITE_STATES))

    for h in range(NUM_HEALTH):
        for u in range(NUM_URGENCY):
            s = encode_state(h, u)
            for hi_obs in range(NUM_HI_OBS):
                for task_obs in range(NUM_TASK_OBS):
                    o = encode_observation(hi_obs, task_obs)
                    A_flat[o, s] = A1[hi_obs, h, u] * A2[task_obs, h, u]

    # Normalize columns (each state's observation distribution sums to 1)
    col_sums = A_flat.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    A_flat = A_flat / col_sums

    return A_flat


def build_flat_B_matrix() -> np.ndarray:
    """Build the flattened transition matrix B: shape (12, 12, 6).

    B[s', s, a] = P(next_state=s' | current_state=s, action=a)

    Constructed from per-factor Kronecker product per action, assuming
    independent factor transitions.
    """
    B_factors = aif.build_B_matrix()
    B1 = B_factors[0]  # (NUM_HEALTH, NUM_HEALTH, NUM_ACTIONS)
    B2 = B_factors[1]  # (NUM_URGENCY, NUM_URGENCY, NUM_ACTIONS)

    B_flat = np.zeros((NUM_COMPOSITE_STATES, NUM_COMPOSITE_STATES, NUM_ACTIONS))

    for a in range(NUM_ACTIONS):
        for h in range(NUM_HEALTH):
            for u in range(NUM_URGENCY):
                s = encode_state(h, u)
                for h_next in range(NUM_HEALTH):
                    for u_next in range(NUM_URGENCY):
                        s_next = encode_state(h_next, u_next)
                        B_flat[s_next, s, a] = (
                            B1[h_next, h, a] * B2[u_next, u, a]
                        )

    # Normalize columns per action
    for a in range(NUM_ACTIONS):
        col_sums = B_flat[:, :, a].sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        B_flat[:, :, a] = B_flat[:, :, a] / col_sums

    return B_flat


def build_flat_C_vector(role_preference: str = 'bug_fix') -> np.ndarray:
    """Build the flattened preference vector C: shape (9,).

    C[o] = log-preference for composite observation o, built as the outer
    sum of C1 (HI preferences) and C2 (task outcome preferences).
    """
    C_factors = aif.build_C_matrix(role_preference=role_preference)
    C1 = C_factors[0]  # (NUM_HI_OBS,)
    C2 = C_factors[1]  # (NUM_TASK_OBS,)

    # Outer sum: C_flat[hi, task] = C1[hi] + C2[task]
    C_2d = C1[:, None] + C2[None, :]  # (NUM_HI_OBS, NUM_TASK_OBS)
    return C_2d.flatten()


def build_flat_D_vector(
    health_prior: str = 'uncertain',
    urgency_prior: str = 'uncertain',
) -> np.ndarray:
    """Build the flattened state prior D: shape (12,).

    D[s] = P(initial composite state), built as outer product of D1 and D2.
    """
    D_factors = aif.build_D_matrix(
        health_prior=health_prior, urgency_prior=urgency_prior
    )
    D_flat = np.outer(D_factors[0], D_factors[1]).flatten()
    D_flat = D_flat / D_flat.sum()
    return D_flat


# ===========================================================================
# SustainHubGenerativeModel
# ===========================================================================

class SustainHubGenerativeModel:
    """Builds a PGMax-compatible GenerativeModel from SustainHub's POMDP.

    Wraps PGMax's GenerativeModel with domain-specific construction logic.
    If PGMax is not available, stores the raw matrices for the NumPy fallback.

    The GenerativeModel receives the matrices in PGMax's list-of-arrays format:
        A: [array(9, 12)]  -- 1 observation modality (flattened)
        B: [array(12, 12, 6)]  -- 1 state factor (flattened)
        C: [array(9,)]  -- preferences over observations
        D: [array(12,)]  -- prior over states
        E: array(6,)  -- policy prior (one per action for T=1)
    """

    def __init__(
        self,
        role: str = 'contributor',
        health_prior: str = 'uncertain',
        urgency_prior: str = 'uncertain',
        habit_strength: float = 1.0,
        planning_horizon: int = 1,
    ):
        self.role = role
        self.num_states = NUM_COMPOSITE_STATES   # 12
        self.num_obs = NUM_COMPOSITE_OBS         # 9
        self.num_actions = NUM_ACTIONS            # 6

        # Build flattened matrices
        self.A = build_flat_A_matrix()          # (9, 12)
        self.B = build_flat_B_matrix()          # (12, 12, 6)
        self.C = build_flat_C_vector(role)      # (9,)
        self.D = build_flat_D_vector(
            health_prior, urgency_prior
        )                                        # (12,)
        self.E = aif.build_E_matrix(
            role=role, habit_strength=habit_strength
        )                                        # (6,)

        # Build GenerativeModel if ALF or PGMax available
        # Both expect lists of arrays for A, B, C, D.
        self.gm = None
        if (_ALF_AVAILABLE or _PGMAX_AVAILABLE) and GenerativeModel is not None:
            try:
                self.gm = GenerativeModel(
                    A=[self.A],
                    B=[self.B],
                    C=[self.C],
                    D=[self.D],
                    E=self.E,
                    T=planning_horizon,
                )
            except Exception as e:
                warnings.warn(
                    f'Failed to build GenerativeModel: {e}. '
                    f'Falling back to NumPy-only mode.'
                )
        # Backwards compatibility
        self.pgmax_model = self.gm

    @property
    def uses_pgmax(self) -> bool:
        """Whether this model is backed by a live GenerativeModel (ALF or PGMax)."""
        return self.gm is not None

    def get_factored_beliefs(
        self, composite_beliefs: np.ndarray
    ) -> list[np.ndarray]:
        """Marginalize composite beliefs back into per-factor beliefs.

        Args:
            composite_beliefs: Array of shape (NUM_COMPOSITE_STATES,).

        Returns:
            [health_beliefs(3,), urgency_beliefs(4,)]
        """
        return get_factored_beliefs(composite_beliefs)


# ===========================================================================
# NumPy fallback inference (used when PGMax is unavailable)
# ===========================================================================

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _numpy_update_beliefs(
    A: np.ndarray,
    beliefs: np.ndarray,
    obs_idx: int,
    num_iterations: int = 16,
) -> np.ndarray:
    """Update composite beliefs given a composite observation (NumPy).

    Simple iterative Bayesian update: Q(s) proportional to P(o|s) * Q_prior(s).
    """
    for _ in range(num_iterations):
        likelihood = np.clip(A[obs_idx, :], 1e-16, None)
        log_posterior = (
            np.log(likelihood) + np.log(np.clip(beliefs, 1e-16, None))
        )
        beliefs = _softmax(log_posterior)
    return beliefs


def _numpy_compute_efe(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    beliefs: np.ndarray,
    action_idx: int,
    gamma: float = 1.0,
) -> float:
    """Compute Expected Free Energy for one action (NumPy).

    G(a) = -gamma * pragmatic - epistemic
    pragmatic = E_Q'[ln P_C(o')]
    epistemic = -E_Q'[H[P(o'|s')]]
    """
    # Predicted next state
    predicted_states = B[:, :, action_idx] @ beliefs
    predicted_states = np.clip(predicted_states, 1e-16, None)
    predicted_states = predicted_states / predicted_states.sum()

    # Predicted observations
    predicted_obs = A @ predicted_states
    predicted_obs = np.clip(predicted_obs, 1e-16, None)
    predicted_obs = predicted_obs / predicted_obs.sum()

    # Pragmatic value: E_Q(o')[ln P_C(o')]
    C_shifted = C - np.max(C)
    log_prefs = C_shifted - np.log(np.sum(np.exp(C_shifted)))
    pragmatic = np.sum(predicted_obs * log_prefs)

    # Epistemic value: -E_Q(s')[H[P(o'|s')]]
    entropy = 0.0
    for s in range(A.shape[1]):
        obs_given_s = np.clip(A[:, s], 1e-16, None)
        entropy -= predicted_states[s] * np.sum(
            obs_given_s * np.log(obs_given_s)
        )

    return float(-gamma * pragmatic - entropy)


def _numpy_select_action(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    E: np.ndarray,
    beliefs: np.ndarray,
    gamma: float = 1.0,
    alpha: float = 16.0,
    rng: np.random.RandomState | None = None,
) -> tuple[int, np.ndarray]:
    """Select action by minimizing EFE (NumPy fallback)."""
    G = np.zeros(NUM_ACTIONS)
    for a in range(NUM_ACTIONS):
        G[a] = _numpy_compute_efe(A, B, C, beliefs, a, gamma)

    log_policy = -G + np.log(np.clip(E, 1e-16, None))
    action_probs = _softmax(alpha * log_policy)

    if rng is not None:
        action_idx = rng.choice(NUM_ACTIONS, p=action_probs)
    else:
        action_idx = np.random.choice(NUM_ACTIONS, p=action_probs)

    return int(action_idx), action_probs


# ===========================================================================
# PGMaxAgent: drop-in replacement for ActiveInferenceAgent
# ===========================================================================

class PGMaxAgent:
    """An active inference agent backed by PGMax (with NumPy fallback).

    Provides the same external interface as
    active_inference.ActiveInferenceAgent so it can be used as a drop-in
    replacement in experiments.py and simulation.py.

    Interface:
        observe(hi_level: str, task_outcome: str) -> None
        decide() -> tuple[str, np.ndarray]
        learn(outcome_valence: float) -> None
        get_state_summary() -> dict[str, Any]

    When PGMax is available, belief updating and policy selection run through
    PGMax's JAX-backed BP engine. Otherwise, equivalent NumPy computations
    are used (producing the same math on the flattened single-factor model).
    """

    def __init__(
        self,
        name: str,
        role: str,
        gamma: float = 1.0,
        alpha: float = 16.0,
        learning_rate: float = 0.1,
        health_prior: str = 'uncertain',
        urgency_prior: str = 'uncertain',
        habit_strength: float = 1.0,
        seed: int = 42,
    ):
        self.name = name
        self.role = role
        self.gamma = gamma
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(seed)

        # Build generative model
        self.gen_model = SustainHubGenerativeModel(
            role=role,
            health_prior=health_prior,
            urgency_prior=urgency_prior,
            habit_strength=habit_strength,
        )

        # Working state
        self.beliefs = self.gen_model.D.copy()    # composite beliefs (12,)
        self.E = self.gen_model.E.copy()           # habit prior (6,)

        # Backend agent (ALF preferred, PGMax fallback)
        self._alf_agent = None
        self._pgmax_agent = None
        if self.gen_model.uses_pgmax and _ALF_AVAILABLE and _ALFAgent is not None:
            try:
                self._alf_agent = _ALFAgent(
                    gm=self.gen_model.gm,
                    gamma=gamma,
                    seed=seed,
                )
            except Exception as e:
                warnings.warn(
                    f'Failed to create ALF agent for {name}: {e}. '
                    f'Trying PGMax fallback.'
                )
        if self._alf_agent is None and self.gen_model.uses_pgmax and _PGMaxAIFAgent is not None:
            try:
                self._pgmax_agent = _PGMaxAIFAgent(
                    gm=self.gen_model.gm,
                    gamma=gamma,
                    learning_rate=learning_rate,
                    bp_iters=16,
                    seed=seed,
                )
            except Exception as e:
                warnings.warn(
                    f'Failed to create PGMax agent for {name}: {e}. '
                    f'Using NumPy fallback.'
                )

        # History (matches ActiveInferenceAgent interface)
        self.action_history: list[str] = []
        self.belief_history: list[list[np.ndarray]] = []
        self.efe_history: list[np.ndarray] = []

    @property
    def uses_pgmax(self) -> bool:
        """Whether this agent is using a JAX backend (ALF or PGMax)."""
        return self._alf_agent is not None or self._pgmax_agent is not None

    @property
    def _active_backend(self) -> str:
        """Return the active backend name."""
        if self._alf_agent is not None:
            return 'alf'
        elif self._pgmax_agent is not None:
            return 'pgmax'
        return 'numpy'

    def observe(self, hi_level: str, task_outcome: str) -> None:
        """Update beliefs based on new observations.

        Args:
            hi_level: One of 'high', 'medium', 'low'.
            task_outcome: One of 'success', 'partial', 'failure'.
        """
        hi_idx = hi_str_to_idx(hi_level)
        task_idx = task_str_to_idx(task_outcome)
        composite_obs_idx = encode_observation(hi_idx, task_idx)

        if self._alf_agent is not None:
            # ALF: analytic Bayesian belief update
            try:
                A = self.gen_model.A
                likelihood = np.clip(A[composite_obs_idx, :], 1e-16, None)
                unnorm = likelihood * self.beliefs
                self.beliefs = unnorm / unnorm.sum()
            except Exception:
                self.beliefs = _numpy_update_beliefs(
                    self.gen_model.A, self.beliefs, composite_obs_idx
                )
        elif self._pgmax_agent is not None:
            try:
                from pgmax.aif import inference as aif_inference
                self._pgmax_agent.beliefs = aif_inference.infer_states(
                    self._pgmax_agent.gm,
                    [composite_obs_idx],
                    prior_beliefs=self._pgmax_agent.beliefs,
                    num_iters=self._pgmax_agent.bp_iters,
                )
                self.beliefs = np.array(
                    self._pgmax_agent.beliefs[0]
                ).flatten()
            except Exception:
                self.beliefs = _numpy_update_beliefs(
                    self.gen_model.A, self.beliefs, composite_obs_idx
                )
        else:
            self.beliefs = _numpy_update_beliefs(
                self.gen_model.A, self.beliefs, composite_obs_idx
            )

        factored = get_factored_beliefs(self.beliefs)
        self.belief_history.append(factored)

    def decide(self) -> tuple[str, np.ndarray]:
        """Select an action using Expected Free Energy minimization.

        Returns:
            action: The selected action string (e.g. 'bug_fix').
            action_probs: Full probability distribution over all 6 actions.
        """
        if self._alf_agent is not None:
            try:
                # ALF: sequential EFE policy evaluation
                gm = self.gen_model.gm
                G = evaluate_all_policies_sequential(gm, [self.beliefs.copy()])

                # Select policy
                policy_idx, policy_probs = _alf_select_action(
                    G, self.E, self.gamma, rng=self.rng,
                )

                selected_policy = gm.policies[policy_idx]
                action_idx = int(selected_policy[0, 0])

                action_probs = self._policy_probs_to_action_probs(
                    policy_probs, gm.policies
                )

            except Exception:
                action_idx, action_probs = _numpy_select_action(
                    self.gen_model.A, self.gen_model.B, self.gen_model.C,
                    self.E, self.beliefs, self.gamma, self.alpha,
                    rng=self.rng,
                )
        elif self._pgmax_agent is not None:
            try:
                from pgmax.aif import policy as aif_policy

                self._pgmax_agent.beliefs = [self.beliefs.copy()]

                G = aif_policy.evaluate_policies(
                    self._pgmax_agent.gm,
                    self._pgmax_agent.beliefs,
                    num_iters=self._pgmax_agent.bp_iters // 2,
                )

                policy_idx, policy_probs = aif_policy.select_action(
                    G, self._pgmax_agent.E,
                    self.gamma,
                    rng=self._pgmax_agent.rng,
                )

                selected_policy = self._pgmax_agent.gm.policies[policy_idx]
                action_idx = int(selected_policy[0, 0])

                action_probs = self._policy_probs_to_action_probs(
                    policy_probs, self._pgmax_agent.gm.policies
                )

                self._pgmax_agent.action_history.append(action_idx)
                self._pgmax_agent.efe_history.append(G.copy())
                self._pgmax_agent.policy_prob_history.append(
                    policy_probs.copy()
                )

            except Exception:
                action_idx, action_probs = _numpy_select_action(
                    self.gen_model.A, self.gen_model.B, self.gen_model.C,
                    self.E, self.beliefs, self.gamma, self.alpha,
                    rng=self.rng,
                )
        else:
            action_idx, action_probs = _numpy_select_action(
                self.gen_model.A, self.gen_model.B, self.gen_model.C,
                self.E, self.beliefs, self.gamma, self.alpha,
                rng=self.rng,
            )

        action = action_idx_to_str(action_idx)
        self.action_history.append(action)
        self.efe_history.append(action_probs.copy())

        return action, action_probs

    def learn(self, outcome_valence: float) -> None:
        """Update habits based on sprint outcome.

        Args:
            outcome_valence: How good the outcome was, in [-1, +1].
        """
        if not self.action_history:
            return

        last_action_idx = action_str_to_idx(self.action_history[-1])
        self.E = aif.update_habits(
            self.E, last_action_idx, outcome_valence, self.learning_rate
        )

        # Sync habits into PGMax agent if present
        if self._pgmax_agent is not None:
            self._pgmax_agent.E = self.E.copy()

    def get_state_summary(self) -> dict[str, Any]:
        """Return a summary of the agent's internal state for logging.

        Compatible with active_inference.ActiveInferenceAgent.get_state_summary().
        """
        factored = get_factored_beliefs(self.beliefs)

        return {
            'name': self.name,
            'role': self.role,
            'backend': self._active_backend,
            'beliefs_health': {
                PROJECT_HEALTH_STATES[i]: float(factored[0][i])
                for i in range(NUM_HEALTH)
            },
            'beliefs_urgency': {
                TASK_URGENCY_STATES[i]: float(factored[1][i])
                for i in range(NUM_URGENCY)
            },
            'habits': {
                ACTIONS[i]: float(self.E[i])
                for i in range(NUM_ACTIONS)
            },
            'gamma': self.gamma,
            'alpha': self.alpha,
            'num_composite_states': self.gen_model.num_states,
            'num_composite_obs': self.gen_model.num_obs,
        }

    def get_factored_beliefs(self) -> list[np.ndarray]:
        """Get beliefs marginalized back into the original factor structure.

        Returns:
            [health_beliefs(3,), urgency_beliefs(4,)]
        """
        return get_factored_beliefs(self.beliefs)

    # --- Internal helpers -------------------------------------------------

    @staticmethod
    def _policy_probs_to_action_probs(
        policy_probs: np.ndarray,
        policies: np.ndarray,
    ) -> np.ndarray:
        """Aggregate policy probabilities into per-action probabilities.

        For T=1 single-factor, each policy maps to exactly one action.
        For T>1, we marginalize: P(a) = sum_{pi: pi[0]==a} P(pi).
        """
        action_probs = np.zeros(NUM_ACTIONS)
        for pi_idx, prob in enumerate(policy_probs):
            first_action = int(policies[pi_idx][0, 0])
            action_probs[first_action] += prob
        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        return action_probs


# ===========================================================================
# Convenience: format AIF context for LLM (same interface as hand-rolled)
# ===========================================================================

def format_aif_context_for_llm(agent: PGMaxAgent) -> str:
    """Format PGMaxAgent state as context for LLM prompting.

    Same output format as active_inference.format_aif_context_for_llm().
    """
    state = agent.get_state_summary()
    health_beliefs = state['beliefs_health']
    urgency_beliefs = state['beliefs_urgency']
    habits = state['habits']

    top_health = max(health_beliefs, key=health_beliefs.get)
    top_urgency = max(urgency_beliefs, key=urgency_beliefs.get)

    sorted_actions = sorted(habits.items(), key=lambda x: x[1], reverse=True)
    action_lines = [f'  {a}: {p:.0%}' for a, p in sorted_actions[:3]]

    backend = state.get('backend', 'unknown')

    return (
        f'[Internal Assessment ({backend})]\n'
        f'Project health belief: {top_health} '
        f'({health_beliefs[top_health]:.0%} confidence)\n'
        f'Current urgency: {top_urgency} '
        f'({urgency_beliefs[top_urgency]:.0%} confidence)\n'
        f'Learned task preferences:\n' + '\n'.join(action_lines)
    )
