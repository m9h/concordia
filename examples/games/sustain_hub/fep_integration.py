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

"""FEP integration layer for SustainHub Active Inference agents.

Bridges the ActiveInferenceAgent POMDP machinery with higher-level
components suitable for LLM prompt injection and Concordia entity
composition.  The two main public classes are:

    FEPAgentComponent
        Wraps ActiveInferenceAgent and exposes observation processing,
        EFE computation / decomposition, action recommendation, and a
        natural-language ``get_fep_context()`` method that can be
        prepended to an LLM decision prompt.

    FEPBeliefTracker
        Maintains a rolling window of Bayesian surprise values and
        detects when the agent's generative model is persistently
        surprised (i.e. the world has changed).

    AdaptiveGenerativeModel
        Extends the standard generative model with the ability to
        re-learn its A and B matrices when surprise is persistently
        high.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from examples.games.sustain_hub import active_inference as aif


# =========================================================================
# FEPBeliefTracker
# =========================================================================


class FEPBeliefTracker:
    """Track Bayesian surprise over time and detect model mismatch.

    Args:
        surprise_threshold: A surprise value (in nats) above which an
            observation is flagged as surprising.
        window_size: Number of recent surprise values to keep for
            rolling statistics.
    """

    def __init__(
        self,
        surprise_threshold: float = 3.0,
        window_size: int = 20,
    ):
        self.surprise_threshold = surprise_threshold
        self.window_size = window_size
        self.surprises: list[float] = []

    def record(self, surprise: float) -> bool:
        """Record a surprise value and return whether it is surprising."""
        self.surprises.append(surprise)
        if len(self.surprises) > self.window_size:
            self.surprises = self.surprises[-self.window_size:]
        return surprise > self.surprise_threshold

    @property
    def mean_surprise(self) -> float:
        if not self.surprises:
            return 0.0
        return float(np.mean(self.surprises))

    @property
    def is_persistently_surprised(self) -> bool:
        """True when recent surprises consistently exceed the threshold."""
        if len(self.surprises) < 3:
            return False
        recent = self.surprises[-3:]
        return all(s > self.surprise_threshold for s in recent)


# =========================================================================
# AdaptiveGenerativeModel
# =========================================================================


class AdaptiveGenerativeModel:
    """A generative model that can re-learn its A / B matrices.

    When a :class:`FEPBeliefTracker` detects persistent surprise the
    agent should call :meth:`adapt` to nudge its likelihood (A) and
    transition (B) matrices toward the recent observation statistics.

    This is a lightweight stand-in for full structure-learning; it
    shifts the existing matrices toward observed frequencies using an
    exponential moving average.

    Args:
        A: Likelihood matrices (list of ndarrays).
        B: Transition matrices (list of ndarrays).
        adaptation_rate: How quickly the model adapts (0-1). Higher
            values mean faster adaptation but less stability.
    """

    def __init__(
        self,
        A: list[np.ndarray],
        B: list[np.ndarray],
        adaptation_rate: float = 0.05,
    ):
        self.A = [a.copy() for a in A]
        self.B = [b.copy() for b in B]
        self.adaptation_rate = adaptation_rate

        # Accumulate observation counts for A-matrix adaptation
        self._obs_counts: list[np.ndarray] = [
            np.ones_like(a) * 0.1 for a in self.A
        ]

    def record_observation(
        self,
        obs_indices: list[int],
        health_belief: np.ndarray,
        urgency_belief: np.ndarray,
    ) -> None:
        """Accumulate soft evidence for A-matrix adaptation."""
        for m, obs_idx in enumerate(obs_indices):
            for h in range(health_belief.shape[0]):
                for u in range(urgency_belief.shape[0]):
                    weight = health_belief[h] * urgency_belief[u]
                    self._obs_counts[m][obs_idx, h, u] += weight

    def adapt(self) -> None:
        """Nudge A matrices toward accumulated observation statistics."""
        eta = self.adaptation_rate
        for m in range(len(self.A)):
            # Normalise counts to distributions per (h, u) column
            target = self._obs_counts[m].copy()
            col_sums = target.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
            target = target / col_sums

            self.A[m] = (1.0 - eta) * self.A[m] + eta * target
            # Re-normalise
            col_sums = self.A[m].sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
            self.A[m] = self.A[m] / col_sums

        # Reset counts after adaptation
        self._obs_counts = [np.ones_like(a) * 0.1 for a in self.A]


# =========================================================================
# FEPAgentComponent
# =========================================================================


class FEPAgentComponent:
    """High-level FEP wrapper around :class:`ActiveInferenceAgent`.

    Designed for use as a Concordia entity component or as a standalone
    analysis object (as in the curriculum notebooks).

    Args:
        name: Agent display name.
        role: One of ``contributor``, ``innovator``, ``knowledge_curator``,
            ``maintainer``.
        gamma: EFE precision (inverse temperature).
        alpha: Action selection precision.
        adaptive: Whether to use an :class:`AdaptiveGenerativeModel` that
            re-learns under persistent surprise.
        surprise_threshold: Threshold (nats) for the belief tracker.
        learning_rate: Habit learning rate forwarded to the inner agent.
    """

    def __init__(
        self,
        name: str,
        role: str = 'contributor',
        gamma: float = 1.0,
        alpha: float = 16.0,
        adaptive: bool = False,
        surprise_threshold: float = 3.0,
        learning_rate: float = 0.1,
    ):
        self.name = name
        self.role = role
        self.adaptive = adaptive

        # Inner AIF agent
        self._agent = aif.ActiveInferenceAgent(
            name=name,
            role=role,
            gamma=gamma,
            alpha=alpha,
            learning_rate=learning_rate,
            health_prior='uncertain',
        )

        # Belief tracker
        self._tracker = FEPBeliefTracker(
            surprise_threshold=surprise_threshold,
        )

        # Adaptive model (optional)
        self._adaptive_model: AdaptiveGenerativeModel | None = None
        if adaptive:
            self._adaptive_model = AdaptiveGenerativeModel(
                A=self._agent.A,
                B=self._agent.B,
            )

        # Cache for last observation result
        self._last_obs_result: dict[str, Any] = {}

    # ----- properties that expose inner-agent state -----

    @property
    def beliefs(self) -> list[np.ndarray]:
        return self._agent.beliefs

    @property
    def gamma(self) -> float:
        return self._agent.gamma

    @property
    def alpha(self) -> float:
        return self._agent.alpha

    # ----- observation processing -----

    def process_observation(
        self,
        text: str,
        hi_level: str,
        task_outcome: str,
    ) -> dict[str, Any]:
        """Process a new observation and update beliefs.

        Args:
            text: Free-form narrative (currently stored but unused by
                the POMDP; reserved for LLM prompt injection).
            hi_level: One of ``'high'``, ``'medium'``, ``'low'``.
            task_outcome: One of ``'success'``, ``'partial'``, ``'failure'``.

        Returns:
            Dictionary with keys:
                ``obs_idx``        -- composite observation index
                ``surprise``       -- Bayesian surprise in nats
                ``is_surprising``  -- whether surprise exceeds threshold
                ``health_beliefs`` -- posterior over health states (ndarray)
        """
        hi_idx = aif.HI_OBSERVATIONS.index(hi_level)
        task_idx = aif.TASK_OBSERVATIONS.index(task_outcome)
        obs_idx = hi_idx * aif.NUM_TASK_OBS + task_idx

        # --- compute surprise BEFORE updating beliefs ---
        predicted_obs = self._predicted_observation()
        surprise = -float(np.log(np.clip(predicted_obs[hi_idx], 1e-16, None)))
        is_surprising = self._tracker.record(surprise)

        # --- belief update ---
        self._agent.observe(hi_level, task_outcome)

        # --- adaptive model bookkeeping ---
        if self._adaptive_model is not None:
            self._adaptive_model.record_observation(
                [hi_idx, task_idx],
                self._agent.beliefs[0],
                self._agent.beliefs[1],
            )
            if self._tracker.is_persistently_surprised:
                self._adaptive_model.adapt()
                # Push adapted A back into the agent
                self._agent.A = [a.copy() for a in self._adaptive_model.A]

        result: dict[str, Any] = {
            'obs_idx': obs_idx,
            'surprise': surprise,
            'is_surprising': is_surprising,
            'health_beliefs': self._agent.beliefs[0].copy(),
        }
        self._last_obs_result = result
        return result

    # ----- EFE computation -----

    def compute_efe_all_actions(self) -> np.ndarray:
        """Return an array of EFE values, one per action.

        Lower EFE is better (agent minimises expected free energy).
        """
        G = np.zeros(aif.NUM_ACTIONS)
        for a in range(aif.NUM_ACTIONS):
            G[a] = aif.compute_expected_free_energy(
                self._agent.A,
                self._agent.B,
                self._agent.C,
                self._agent.D,
                a,
                self._agent.beliefs,
                self._agent.gamma,
            )
        return G

    def get_recommended_action(self) -> tuple[int, np.ndarray]:
        """Select an action by minimising EFE (via softmax policy).

        Returns:
            (action_idx, action_probabilities)
        """
        action_idx, probs = aif.select_action(
            self._agent.A,
            self._agent.B,
            self._agent.C,
            self._agent.D,
            self._agent.E,
            self._agent.beliefs,
            self._agent.gamma,
            self._agent.alpha,
        )
        return action_idx, probs

    def compute_efe_decomposition(self) -> dict[str, dict[str, float]]:
        """Decompose EFE into pragmatic and epistemic terms per action.

        Returns:
            Mapping ``action_name -> {G_total, pragmatic, epistemic}``.
        """
        decomp: dict[str, dict[str, float]] = {}

        for a_idx, a_name in enumerate(aif.ACTIONS):
            # Predict next state
            predicted_health = (
                self._agent.B[0][:, :, a_idx] @ self._agent.beliefs[0]
            )
            predicted_urgency = (
                self._agent.B[1][:, :, a_idx] @ self._agent.beliefs[1]
            )

            pragmatic_total = 0.0
            epistemic_total = 0.0

            for m, Am in enumerate(self._agent.A):
                Cm = self._agent.C[m]

                # Predicted observations
                predicted_obs = np.zeros(Am.shape[0])
                for h in range(aif.NUM_HEALTH):
                    for u in range(aif.NUM_URGENCY):
                        predicted_obs += (
                            Am[:, h, u]
                            * predicted_health[h]
                            * predicted_urgency[u]
                        )
                predicted_obs = np.clip(predicted_obs, 1e-16, None)
                predicted_obs = predicted_obs / predicted_obs.sum()

                # Pragmatic: E_Q[ln P(o')]
                log_prefs = Cm - _log_sum_exp(Cm)
                pragmatic = float(np.sum(predicted_obs * log_prefs))

                # Epistemic: -E_Q[H[P(o'|s')]]
                entropy = 0.0
                for h in range(aif.NUM_HEALTH):
                    for u in range(aif.NUM_URGENCY):
                        obs_given = np.clip(Am[:, h, u], 1e-16, None)
                        sp = predicted_health[h] * predicted_urgency[u]
                        entropy -= sp * float(
                            np.sum(obs_given * np.log(obs_given))
                        )
                epistemic = entropy

                pragmatic_total += pragmatic
                epistemic_total += epistemic

            gamma = self._agent.gamma
            G_total = -gamma * pragmatic_total - epistemic_total

            decomp[a_name] = {
                'G_total': G_total,
                'pragmatic': pragmatic_total,
                'epistemic': epistemic_total,
            }

        return decomp

    # ----- natural-language context for LLM prompts -----

    def get_fep_context(self) -> str:
        """Return a natural-language summary of the agent's FEP state.

        This string is designed to be prepended to an LLM decision
        prompt so that the language model can reason about beliefs,
        surprise, and action recommendations in natural language.
        """
        # Health beliefs
        hb = self._agent.beliefs[0]
        health_lines = '  '.join(
            f'{aif.PROJECT_HEALTH_STATES[i]}: {hb[i]:.1%}'
            for i in range(aif.NUM_HEALTH)
        )
        top_health_idx = int(np.argmax(hb))
        top_health = aif.PROJECT_HEALTH_STATES[top_health_idx]

        # Urgency beliefs
        ub = self._agent.beliefs[1]
        top_urgency_idx = int(np.argmax(ub))
        top_urgency = aif.TASK_URGENCY_STATES[top_urgency_idx]

        # Recent surprise
        last_surprise = self._last_obs_result.get('surprise', 0.0)
        mean_surprise = self._tracker.mean_surprise
        is_persistent = self._tracker.is_persistently_surprised

        # EFE decomposition
        decomp = self.compute_efe_decomposition()
        sorted_actions = sorted(
            decomp.items(), key=lambda kv: kv[1]['G_total']
        )

        # Recommended action
        rec_idx, rec_probs = self.get_recommended_action()
        rec_action = aif.ACTIONS[rec_idx]

        lines = [
            f'[FEP Context for {self.name} ({self.role})]',
            f'',
            f'Belief state:',
            f'  Project health: {top_health} ({hb[top_health_idx]:.0%} confidence)',
            f'  Health distribution: {health_lines}',
            f'  Current urgency: {top_urgency} ({ub[top_urgency_idx]:.0%})',
            f'',
            f'Surprise:',
            f'  Last observation: {last_surprise:.2f} nats',
            f'  Mean surprise: {mean_surprise:.2f} nats',
            f'  Model mismatch: {"YES -- consider model update" if is_persistent else "no"}',
            f'',
            f'EFE-ranked actions (lower G = better):',
        ]
        for a_name, vals in sorted_actions[:3]:
            lines.append(
                f'  {a_name:<15} G={vals["G_total"]:+.3f}  '
                f'(pragmatic={vals["pragmatic"]:+.3f}, '
                f'epistemic={vals["epistemic"]:+.3f})'
            )
        lines += [
            f'',
            f'Recommended action: {rec_action} '
            f'(prob={rec_probs[rec_idx]:.0%})',
        ]

        return '\n'.join(lines)

    # ----- internal helpers -----

    def _predicted_observation(self) -> np.ndarray:
        """Predicted HI observation distribution given current beliefs."""
        predicted_obs = np.zeros(aif.NUM_HI_OBS)
        for h in range(aif.NUM_HEALTH):
            for u in range(aif.NUM_URGENCY):
                predicted_obs += (
                    self._agent.A[0][:, h, u]
                    * self._agent.beliefs[0][h]
                    * self._agent.beliefs[1][u]
                )
        predicted_obs = np.clip(predicted_obs, 1e-16, None)
        return predicted_obs / predicted_obs.sum()


# =========================================================================
# Utility (mirrors aif._log_sum_exp but kept local to avoid private access)
# =========================================================================


def _log_sum_exp(x: np.ndarray) -> float:
    c = float(np.max(x))
    return c + float(np.log(np.sum(np.exp(x - c))))
