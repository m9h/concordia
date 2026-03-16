"""Concordia-FEP Integration: connects LLM agents to Free Energy Principle dynamics.

Bridges Concordia's LLM-based multi-agent simulation framework with PGMax's
Active Inference engine, enabling agents that combine language model reasoning
with principled Bayesian belief updating and information-seeking behavior.

Architecture:

    Concordia Entity
        |
        +-- FEPAgentComponent (ContextComponent)
        |       |
        |       +-- FEPBeliefTracker (belief/surprise history)
        |       +-- FEPLLMBridge (FEP <-> natural language)
        |       +-- AdaptiveGenerativeModel (self-updating POMDP)
        |
        +-- Other Concordia components (memory, observation, etc.)

The FEPAgentComponent hooks into Concordia's pre_observe / pre_act lifecycle:

    pre_observe: Parse text observation -> POMDP observation index -> belief update
    pre_act:     Compute EFE for all actions -> format FEP context for LLM prompt
    post_act:    Record taken action for learning
    update:      Optionally trigger adaptive model re-learning

All computation uses the NumPy fallback from pgmax_bridge.py, so PGMax/JAX
are NOT required at runtime. When PGMax IS available, the AdaptiveGenerativeModel
can use differentiable learning (learning.py) for model updates.

POMDP dimensions (SustainHub):
    States:       3 health x 4 urgency = 12 composite states
    Observations: 3 HI levels x 3 task outcomes = 9 composite observations
    Actions:      6 (bug_fix, feature, documentation, code_review, mentor, skip)

References:
    Friston, K. (2010). The free-energy principle: a unified brain theory?
    Smith, Friston & Whyte (2022). A Step-by-Step Tutorial on Active Inference.
    Parr, Pezzulo & Friston (2022). Active Inference. MIT Press.
"""

import re
from typing import Any, Optional

import numpy as np

from examples.games.sustain_hub import pgmax_bridge

# Re-export domain constants for convenience
PROJECT_HEALTH_STATES = pgmax_bridge.PROJECT_HEALTH_STATES
TASK_URGENCY_STATES = pgmax_bridge.TASK_URGENCY_STATES
HI_OBSERVATIONS = pgmax_bridge.HI_OBSERVATIONS
TASK_OBSERVATIONS = pgmax_bridge.TASK_OBSERVATIONS
ACTIONS = pgmax_bridge.ACTIONS

NUM_HEALTH = pgmax_bridge.NUM_HEALTH
NUM_URGENCY = pgmax_bridge.NUM_URGENCY
NUM_HI_OBS = pgmax_bridge.NUM_HI_OBS
NUM_TASK_OBS = pgmax_bridge.NUM_TASK_OBS
NUM_ACTIONS = pgmax_bridge.NUM_ACTIONS
NUM_COMPOSITE_STATES = pgmax_bridge.NUM_COMPOSITE_STATES
NUM_COMPOSITE_OBS = pgmax_bridge.NUM_COMPOSITE_OBS


# ---------------------------------------------------------------------------
# Utility: softmax (local copy to avoid internal import)
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ===========================================================================
# FEPBeliefTracker: tracks beliefs, free energy, and surprise over time
# ===========================================================================

class FEPBeliefTracker:
    """Tracks an agent's beliefs, free energy, and surprise over time.

    Provides the historical record needed for:
    - Monitoring convergence / divergence of beliefs
    - Detecting surprising observations that should trigger model updating
    - Plotting free energy trajectories for analysis

    Attributes:
        belief_history: List of composite belief vectors at each timestep.
        factored_belief_history: List of [health_beliefs, urgency_beliefs].
        vfe_history: Variational Free Energy at each observation.
        efe_history: Expected Free Energy vectors (per action) at each decision.
        surprise_history: Bayesian surprise -log P(o|model) at each observation.
        action_history: Actions taken at each decision point.
    """

    def __init__(self, surprise_threshold: float = 3.0):
        """Initialize the tracker.

        Args:
            surprise_threshold: Observations with surprise exceeding this
                value (in nats) are flagged as "surprising". Default 3.0
                corresponds roughly to P(o) < 0.05.
        """
        self.surprise_threshold = surprise_threshold

        self.belief_history: list[np.ndarray] = []
        self.factored_belief_history: list[list[np.ndarray]] = []
        self.vfe_history: list[float] = []
        self.efe_history: list[np.ndarray] = []
        self.surprise_history: list[float] = []
        self.surprise_flags: list[bool] = []
        self.action_history: list[str] = []
        self.observation_history: list[int] = []

    def record_observation(
        self,
        beliefs: np.ndarray,
        obs_idx: int,
        A: np.ndarray,
    ) -> float:
        """Record a belief update after observing obs_idx.

        Computes and stores:
        - Updated beliefs (composite and factored)
        - Bayesian surprise: -log P(o | model) = -log sum_s A[o,s] * Q(s)
        - VFE: sum_s Q(s) * [log Q(s) - log P(o,s)]

        Args:
            beliefs: Posterior beliefs after incorporating the observation.
            obs_idx: The composite observation index.
            A: Likelihood matrix, shape (num_obs, num_states).

        Returns:
            surprise: The Bayesian surprise in nats.
        """
        self.belief_history.append(beliefs.copy())
        self.factored_belief_history.append(
            pgmax_bridge.get_factored_beliefs(beliefs)
        )
        self.observation_history.append(obs_idx)

        # Bayesian surprise: -log P(o) = -log sum_s P(o|s) * Q_prior(s)
        # We use the *prior* beliefs (before update) if available, otherwise
        # approximate with the posterior. For simplicity, use posterior here
        # (which slightly underestimates surprise).
        marginal_obs = np.clip(A[obs_idx, :] @ beliefs, 1e-16, None)
        surprise = float(-np.log(marginal_obs))
        self.surprise_history.append(surprise)
        self.surprise_flags.append(surprise > self.surprise_threshold)

        # Variational Free Energy: F = sum_s Q(s) [log Q(s) - log P(o,s|model)]
        # where P(o,s) = P(o|s) * P(s) and we use Q(s) as approximate P(s)
        log_q = np.log(np.clip(beliefs, 1e-16, None))
        log_joint = np.log(np.clip(A[obs_idx, :], 1e-16, None)) + log_q
        vfe = float(np.sum(beliefs * (log_q - log_joint)))
        self.vfe_history.append(vfe)

        return surprise

    def record_decision(
        self,
        efe_values: np.ndarray,
        action: str,
    ) -> None:
        """Record a decision (EFE evaluation + selected action).

        Args:
            efe_values: EFE for each action, shape (num_actions,).
            action: The action string that was selected.
        """
        self.efe_history.append(efe_values.copy())
        self.action_history.append(action)

    def is_surprising(self, lookback: int = 1) -> bool:
        """Check if recent observations were surprising.

        Args:
            lookback: Number of recent observations to check.

        Returns:
            True if any of the last `lookback` observations exceeded
            the surprise threshold.
        """
        if not self.surprise_flags:
            return False
        return any(self.surprise_flags[-lookback:])

    def mean_recent_surprise(self, window: int = 5) -> float:
        """Compute mean surprise over a recent window.

        Args:
            window: Number of recent observations to average.

        Returns:
            Mean surprise in nats, or 0.0 if no observations yet.
        """
        if not self.surprise_history:
            return 0.0
        recent = self.surprise_history[-window:]
        return float(np.mean(recent))

    def mean_recent_vfe(self, window: int = 5) -> float:
        """Compute mean VFE over a recent window.

        Args:
            window: Number of recent observations to average.

        Returns:
            Mean VFE, or 0.0 if no observations yet.
        """
        if not self.vfe_history:
            return 0.0
        recent = self.vfe_history[-window:]
        return float(np.mean(recent))

    @property
    def num_observations(self) -> int:
        """Number of observations recorded."""
        return len(self.observation_history)

    @property
    def num_decisions(self) -> int:
        """Number of decisions recorded."""
        return len(self.action_history)


# ===========================================================================
# FEPLLMBridge: translates between FEP quantities and natural language
# ===========================================================================

class FEPLLMBridge:
    """Bridges FEP dynamics and LLM prompting.

    Translates quantitative FEP metrics (beliefs, VFE, EFE decomposition,
    epistemic/pragmatic drives) into natural-language context strings that
    an LLM can use for decision-making.

    Also handles the reverse: interpreting LLM text output as POMDP action
    indices and parsing observations from Concordia text events.
    """

    # Keyword patterns for parsing observations from Concordia text
    _HI_PATTERNS = {
        'high': re.compile(
            r'(?:harmony|health|HI).{0,30}(?:high|excellent|strong|thriving)',
            re.IGNORECASE,
        ),
        'medium': re.compile(
            r'(?:harmony|health|HI).{0,30}(?:medium|moderate|average|okay|stable)',
            re.IGNORECASE,
        ),
        'low': re.compile(
            r'(?:harmony|health|HI).{0,30}(?:low|poor|weak|declining|critical)',
            re.IGNORECASE,
        ),
    }

    _TASK_PATTERNS = {
        'success': re.compile(
            r'(?:task|result|outcome).{0,30}(?:success|complete|resolved|fixed|done)',
            re.IGNORECASE,
        ),
        'partial': re.compile(
            r'(?:task|result|outcome).{0,30}(?:partial|mixed|some|progress)',
            re.IGNORECASE,
        ),
        'failure': re.compile(
            r'(?:task|result|outcome).{0,30}(?:fail|fail(?:ed|ure)|broke|regression|worse)',
            re.IGNORECASE,
        ),
    }

    # Action keyword patterns for interpreting LLM output
    _ACTION_PATTERNS = {
        'bug_fix': re.compile(r'bug.?fix|fix.?bug|resolve.?bug|debug', re.IGNORECASE),
        'feature': re.compile(r'feature|new.?feature|implement|develop', re.IGNORECASE),
        'documentation': re.compile(r'document|docs|write.?doc|update.?doc', re.IGNORECASE),
        'code_review': re.compile(r'code.?review|review.?code|review.?pr|pull.?request', re.IGNORECASE),
        'mentor': re.compile(r'mentor|teach|guide|help.?team|onboard', re.IGNORECASE),
        'skip': re.compile(r'skip|pass|nothing|idle|wait', re.IGNORECASE),
    }

    @staticmethod
    def format_beliefs_for_llm(
        beliefs: np.ndarray,
        vfe: float,
        efe_values: Optional[np.ndarray] = None,
    ) -> str:
        """Format current beliefs and free energy into LLM prompt context.

        Args:
            beliefs: Composite belief vector, shape (12,).
            vfe: Current Variational Free Energy.
            efe_values: Optional EFE per action, shape (6,).

        Returns:
            Multi-line string describing the agent's internal FEP state.
        """
        factored = pgmax_bridge.get_factored_beliefs(beliefs)
        health_beliefs = factored[0]
        urgency_beliefs = factored[1]

        # Find MAP estimates
        map_health = PROJECT_HEALTH_STATES[int(np.argmax(health_beliefs))]
        map_urgency = TASK_URGENCY_STATES[int(np.argmax(urgency_beliefs))]
        health_conf = float(np.max(health_beliefs))
        urgency_conf = float(np.max(urgency_beliefs))

        # Belief entropy (uncertainty measure)
        health_entropy = float(-np.sum(
            health_beliefs * np.log(np.clip(health_beliefs, 1e-16, None))
        ))
        urgency_entropy = float(-np.sum(
            urgency_beliefs * np.log(np.clip(urgency_beliefs, 1e-16, None))
        ))

        lines = [
            '[FEP Internal State]',
            f'Believed project health: {map_health} '
            f'(confidence: {health_conf:.0%}, uncertainty: {health_entropy:.2f} nats)',
            f'Believed task urgency: {map_urgency} '
            f'(confidence: {urgency_conf:.0%}, uncertainty: {urgency_entropy:.2f} nats)',
            f'Model fit (VFE): {vfe:.3f} '
            f'({"good fit" if vfe < 0.5 else "poor fit, model may need updating"})',
        ]

        # Per-state breakdown
        health_detail = ', '.join(
            f'{PROJECT_HEALTH_STATES[i]}: {health_beliefs[i]:.0%}'
            for i in range(NUM_HEALTH)
        )
        urgency_detail = ', '.join(
            f'{TASK_URGENCY_STATES[i]}: {urgency_beliefs[i]:.0%}'
            for i in range(NUM_URGENCY)
        )
        lines.append(f'Health distribution: [{health_detail}]')
        lines.append(f'Urgency distribution: [{urgency_detail}]')

        if efe_values is not None:
            lines.append('Expected Free Energy per action (lower = better):')
            sorted_actions = sorted(
                zip(ACTIONS, efe_values), key=lambda x: x[1]
            )
            for action, efe in sorted_actions:
                lines.append(f'  {action}: {efe:.3f}')

        return '\n'.join(lines)

    @staticmethod
    def format_epistemic_drive(
        epistemic_value: float,
        pragmatic_value: float,
    ) -> str:
        """Format epistemic vs pragmatic drive balance as natural language.

        Args:
            epistemic_value: Information-seeking drive (negative = stronger).
            pragmatic_value: Reward-seeking drive (negative = stronger).

        Returns:
            String describing the agent's motivational balance.
        """
        total = abs(epistemic_value) + abs(pragmatic_value)
        if total < 1e-8:
            return (
                'Motivational balance: Neither information-seeking nor '
                'reward-seeking drives are active. The agent is in equilibrium.'
            )

        epist_frac = abs(epistemic_value) / total
        prag_frac = abs(pragmatic_value) / total

        if epist_frac > 0.7:
            drive_desc = (
                'The agent is primarily driven by CURIOSITY / INFORMATION-SEEKING. '
                'It wants to reduce uncertainty about the project state before '
                'committing to a specific action strategy.'
            )
        elif prag_frac > 0.7:
            drive_desc = (
                'The agent is primarily driven by GOAL-SEEKING / EXPLOITATION. '
                'It is relatively confident about the project state and wants to '
                'take actions that will lead to preferred outcomes.'
            )
        else:
            drive_desc = (
                'The agent balances CURIOSITY and GOAL-SEEKING. '
                'It seeks both information about the project state and '
                'actions that lead to good outcomes.'
            )

        return (
            f'Motivational balance: '
            f'epistemic (info-seeking): {epist_frac:.0%}, '
            f'pragmatic (goal-seeking): {prag_frac:.0%}.\n'
            f'{drive_desc}'
        )

    @staticmethod
    def interpret_llm_action(
        llm_output: str,
        available_actions: Optional[list[str]] = None,
    ) -> int:
        """Interpret LLM text output as a POMDP action index.

        Uses pattern matching to find which action the LLM is recommending.
        Falls back to the first available action if no match is found.

        Args:
            llm_output: The LLM's text response.
            available_actions: List of valid action strings. Defaults to ACTIONS.

        Returns:
            Integer action index into ACTIONS.
        """
        if available_actions is None:
            available_actions = list(ACTIONS)

        # Try exact match first (case-insensitive)
        llm_lower = llm_output.lower().strip()
        for i, action in enumerate(ACTIONS):
            if action.lower() == llm_lower:
                if action in available_actions:
                    return i

        # Try pattern matching
        scores = {}
        for action, pattern in FEPLLMBridge._ACTION_PATTERNS.items():
            if action in available_actions:
                matches = pattern.findall(llm_output)
                if matches:
                    scores[action] = len(matches)

        if scores:
            best_action = max(scores, key=scores.get)
            return ACTIONS.index(best_action)

        # Try substring match
        for i, action in enumerate(ACTIONS):
            if action.replace('_', ' ') in llm_lower or action in llm_lower:
                if action in available_actions:
                    return i

        # Fallback: first available action
        for i, action in enumerate(ACTIONS):
            if action in available_actions:
                return i

        return 0

    @staticmethod
    def parse_observation_from_text(text: str) -> tuple[str, str]:
        """Parse a Concordia text observation into POMDP observation components.

        Extracts harmony index level and task outcome from natural language text.

        Args:
            text: The observation text from Concordia.

        Returns:
            Tuple of (hi_level, task_outcome) strings.
            Defaults to ('medium', 'partial') if parsing fails.
        """
        hi_level = 'medium'  # default
        task_outcome = 'partial'  # default

        for level, pattern in FEPLLMBridge._HI_PATTERNS.items():
            if pattern.search(text):
                hi_level = level
                break

        for outcome, pattern in FEPLLMBridge._TASK_PATTERNS.items():
            if pattern.search(text):
                task_outcome = outcome
                break

        return hi_level, task_outcome


# ===========================================================================
# AdaptiveGenerativeModel: self-updating POMDP model
# ===========================================================================

class AdaptiveGenerativeModel:
    """A generative model that updates itself based on accumulated experience.

    Wraps SustainHubGenerativeModel and periodically re-learns the A (likelihood)
    and B (transition) matrices from observation-action data when the model's
    predictions diverge from reality (detected via rising VFE/surprise).

    When PGMax with JAX is available, uses differentiable learning (learning.py).
    Otherwise, uses a simple Bayesian count-based update as a numpy fallback.

    Attributes:
        gen_model: The underlying SustainHubGenerativeModel.
        relearn_threshold: Mean surprise above which re-learning is triggered.
        min_data_for_relearn: Minimum observations before re-learning is allowed.
        relearn_count: Number of times the model has been re-learned.
    """

    def __init__(
        self,
        role: str = 'contributor',
        health_prior: str = 'uncertain',
        urgency_prior: str = 'uncertain',
        relearn_threshold: float = 2.5,
        min_data_for_relearn: int = 10,
        learning_rate: float = 0.01,
        relearn_epochs: int = 50,
    ):
        """Initialize the adaptive model.

        Args:
            role: Agent role for preference construction.
            health_prior: Initial health prior type.
            urgency_prior: Initial urgency prior type.
            relearn_threshold: Mean surprise (nats) above which re-learning
                is triggered.
            min_data_for_relearn: Minimum number of observations before
                re-learning can be triggered.
            learning_rate: Learning rate for gradient-based re-learning.
            relearn_epochs: Number of epochs for gradient-based re-learning.
        """
        self.gen_model = pgmax_bridge.SustainHubGenerativeModel(
            role=role,
            health_prior=health_prior,
            urgency_prior=urgency_prior,
        )
        self.relearn_threshold = relearn_threshold
        self.min_data_for_relearn = min_data_for_relearn
        self.learning_rate = learning_rate
        self.relearn_epochs = relearn_epochs
        self.relearn_count = 0

        # Accumulated data for re-learning
        self._obs_data: list[int] = []
        self._action_data: list[int] = []

        # Original matrices (for comparison / reset)
        self._original_A = self.gen_model.A.copy()
        self._original_B = self.gen_model.B.copy()

    @property
    def A(self) -> np.ndarray:
        return self.gen_model.A

    @property
    def B(self) -> np.ndarray:
        return self.gen_model.B

    @property
    def C(self) -> np.ndarray:
        return self.gen_model.C

    @property
    def D(self) -> np.ndarray:
        return self.gen_model.D

    @property
    def E(self) -> np.ndarray:
        return self.gen_model.E

    def record_transition(self, obs_idx: int, action_idx: int) -> None:
        """Record an observation-action pair for future re-learning.

        Args:
            obs_idx: Composite observation index.
            action_idx: Action index that was taken.
        """
        self._obs_data.append(obs_idx)
        self._action_data.append(action_idx)

    def should_relearn(self, tracker: 'FEPBeliefTracker') -> bool:
        """Check whether re-learning should be triggered.

        Conditions:
        1. Enough data has been accumulated (>= min_data_for_relearn)
        2. Mean recent surprise exceeds the threshold

        Args:
            tracker: The belief tracker with surprise history.

        Returns:
            True if re-learning should be triggered.
        """
        if len(self._obs_data) < self.min_data_for_relearn:
            return False
        return tracker.mean_recent_surprise() > self.relearn_threshold

    def relearn(self) -> bool:
        """Re-learn A and B matrices from accumulated data.

        Attempts to use PGMax's differentiable learning (JAX-backed) first.
        Falls back to a simple numpy count-based update if JAX is unavailable.

        Returns:
            True if re-learning succeeded, False otherwise.
        """
        if len(self._obs_data) < 2:
            return False

        # Try PGMax differentiable learning
        try:
            from pgmax.aif.learning import learn_model as pgmax_learn_model
            import jax.numpy as jnp  # noqa: F401 -- proves JAX is available

            obs_arr = np.array(self._obs_data)
            act_arr = np.array(self._action_data)
            # Pad actions to match observations length
            if len(act_arr) < len(obs_arr):
                act_arr = np.concatenate([act_arr, [0] * (len(obs_arr) - len(act_arr))])

            result = pgmax_learn_model(
                observations=obs_arr,
                actions=act_arr,
                num_obs=NUM_COMPOSITE_OBS,
                num_states=NUM_COMPOSITE_STATES,
                num_actions=NUM_ACTIONS,
                D=self.gen_model.D,
                C=self.gen_model.C,
                num_epochs=self.relearn_epochs,
                lr=self.learning_rate,
            )
            self.gen_model.A = result.learned_A[0]
            self.gen_model.B = result.learned_B[0]
            self.relearn_count += 1
            return True

        except (ImportError, Exception):
            pass

        # Numpy fallback: count-based Bayesian update
        return self._numpy_relearn()

    def _numpy_relearn(self) -> bool:
        """Fallback re-learning using observation counts.

        Updates A matrix columns by blending prior with observed frequencies.
        This is a simplified Bayesian update using pseudo-counts.

        Returns:
            True if re-learning succeeded.
        """
        if len(self._obs_data) < 2:
            return False

        # Count observation frequencies per (implicit) state
        # Since we don't observe states directly, we use a simple heuristic:
        # weight each observation by the posterior belief at that time.
        # For now, just update A based on raw observation frequencies.
        obs_counts = np.zeros(NUM_COMPOSITE_OBS)
        for o in self._obs_data:
            obs_counts[o] += 1

        # Blend with prior A (column-wise average)
        total_obs = obs_counts.sum()
        if total_obs > 0:
            obs_freq = obs_counts / total_obs
            # Use a conservative blend: 80% original, 20% empirical
            blend_weight = min(0.2, len(self._obs_data) / 100.0)
            for s in range(NUM_COMPOSITE_STATES):
                blended = (1 - blend_weight) * self._original_A[:, s] + blend_weight * obs_freq
                blended = np.clip(blended, 1e-8, None)
                self.gen_model.A[:, s] = blended / blended.sum()

        self.relearn_count += 1
        return True

    def model_divergence(self) -> float:
        """Compute KL divergence between current and original A matrices.

        Returns:
            Mean KL divergence across columns (states).
        """
        kl_total = 0.0
        for s in range(NUM_COMPOSITE_STATES):
            p = np.clip(self.gen_model.A[:, s], 1e-16, None)
            q = np.clip(self._original_A[:, s], 1e-16, None)
            kl_total += float(np.sum(p * np.log(p / q)))
        return kl_total / NUM_COMPOSITE_STATES

    def reset_to_original(self) -> None:
        """Reset A and B matrices to their original values."""
        self.gen_model.A = self._original_A.copy()
        self.gen_model.B = self._original_B.copy()


# ===========================================================================
# FEPAgentComponent: Concordia ContextComponent wrapping PGMax AIF
# ===========================================================================

class FEPAgentComponent:
    """A Concordia-compatible component that wraps PGMax Active Inference.

    This component can be used as a context provider for Concordia agents.
    It hooks into the observation/action lifecycle:

    - pre_observe: Parses text observations into POMDP observation indices,
      updates beliefs via Bayesian inference, records surprise.
    - pre_act: Computes EFE for all actions, formats FEP context as a
      text string that gets injected into the LLM prompt.
    - post_act: Records the action taken for learning.
    - update: Checks if adaptive re-learning should be triggered.

    The component does NOT make decisions on its own. Instead, it provides
    "FEP context" to the LLM, enriching the prompt with:
    - Current beliefs about project health and urgency
    - Free energy metrics (model fit quality)
    - Epistemic vs pragmatic drive balance
    - Ranked action recommendations from EFE minimization

    This creates a hybrid agent: LLM reasoning + FEP Bayesian rationality.

    Usage:
        component = FEPAgentComponent(
            name='fep_advisor',
            role='contributor',
            pre_act_label='FEP Assessment',
        )
        # Can be used standalone (without Concordia entity) via:
        component.process_observation('The harmony index is low...')
        context = component.get_fep_context()
        action_idx = component.get_recommended_action()
    """

    def __init__(
        self,
        name: str = 'fep_advisor',
        role: str = 'contributor',
        gamma: float = 1.0,
        alpha: float = 16.0,
        learning_rate: float = 0.1,
        health_prior: str = 'uncertain',
        urgency_prior: str = 'uncertain',
        surprise_threshold: float = 3.0,
        adaptive: bool = True,
        relearn_threshold: float = 2.5,
        min_data_for_relearn: int = 10,
        pre_act_label: str = 'FEP Assessment',
        seed: int = 42,
    ):
        """Initialize the FEP agent component.

        Args:
            name: Component name.
            role: Agent role (contributor, innovator, etc.).
            gamma: EFE precision (inverse temperature).
            alpha: Action selection precision.
            learning_rate: Habit learning rate.
            health_prior: Initial health prior type.
            urgency_prior: Initial urgency prior type.
            surprise_threshold: Surprise threshold for flagging (nats).
            adaptive: Whether to use adaptive model re-learning.
            relearn_threshold: Surprise threshold for triggering re-learning.
            min_data_for_relearn: Minimum data points before re-learning.
            pre_act_label: Label prefix for pre_act output.
            seed: Random seed.
        """
        self.name = name
        self.role = role
        self.gamma = gamma
        self.alpha = alpha
        self.learning_rate = learning_rate
        self._pre_act_label = pre_act_label
        self.rng = np.random.RandomState(seed)

        # Core components
        self.adaptive_model = AdaptiveGenerativeModel(
            role=role,
            health_prior=health_prior,
            urgency_prior=urgency_prior,
            relearn_threshold=relearn_threshold,
            min_data_for_relearn=min_data_for_relearn,
        )
        self.tracker = FEPBeliefTracker(surprise_threshold=surprise_threshold)
        self.bridge = FEPLLMBridge()

        # Working state
        self.beliefs = self.adaptive_model.D.copy()
        self.E = self.adaptive_model.E.copy()
        self._last_efe_values: Optional[np.ndarray] = None
        self._last_efe_decomposition: Optional[dict] = None
        self._adaptive = adaptive

    @property
    def A(self) -> np.ndarray:
        return self.adaptive_model.A

    @property
    def B(self) -> np.ndarray:
        return self.adaptive_model.B

    @property
    def C(self) -> np.ndarray:
        return self.adaptive_model.C

    @property
    def D(self) -> np.ndarray:
        return self.adaptive_model.D

    # --- Observation processing -------------------------------------------

    def process_observation(
        self,
        text: str,
        hi_level: Optional[str] = None,
        task_outcome: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process an observation (text or structured) and update beliefs.

        Args:
            text: The observation text from Concordia. Used for parsing
                if hi_level/task_outcome are not provided.
            hi_level: Optional explicit HI level ('high', 'medium', 'low').
            task_outcome: Optional explicit task outcome ('success', 'partial', 'failure').

        Returns:
            Dict with keys: hi_level, task_outcome, obs_idx, surprise,
            is_surprising, beliefs.
        """
        if hi_level is None or task_outcome is None:
            parsed_hi, parsed_task = self.bridge.parse_observation_from_text(text)
            if hi_level is None:
                hi_level = parsed_hi
            if task_outcome is None:
                task_outcome = parsed_task

        hi_idx = pgmax_bridge.hi_str_to_idx(hi_level)
        task_idx = pgmax_bridge.task_str_to_idx(task_outcome)
        obs_idx = pgmax_bridge.encode_observation(hi_idx, task_idx)

        # Bayesian belief update
        self.beliefs = pgmax_bridge._numpy_update_beliefs(
            self.A, self.beliefs, obs_idx
        )

        # Record in tracker
        surprise = self.tracker.record_observation(
            self.beliefs, obs_idx, self.A
        )

        return {
            'hi_level': hi_level,
            'task_outcome': task_outcome,
            'obs_idx': obs_idx,
            'surprise': surprise,
            'is_surprising': surprise > self.tracker.surprise_threshold,
            'beliefs': self.beliefs.copy(),
        }

    # --- Action evaluation ------------------------------------------------

    def compute_efe_all_actions(self) -> np.ndarray:
        """Compute Expected Free Energy for all actions.

        Returns:
            Array of shape (6,) with EFE for each action.
        """
        G = np.zeros(NUM_ACTIONS)
        for a in range(NUM_ACTIONS):
            G[a] = pgmax_bridge._numpy_compute_efe(
                self.A, self.B, self.C, self.beliefs, a, self.gamma
            )
        self._last_efe_values = G
        return G

    def compute_efe_decomposition(self) -> dict[str, np.ndarray]:
        """Compute the epistemic/pragmatic decomposition of EFE for all actions.

        Returns:
            Dict with 'epistemic', 'pragmatic', 'total' arrays of shape (6,).
        """
        epistemic = np.zeros(NUM_ACTIONS)
        pragmatic = np.zeros(NUM_ACTIONS)

        for a in range(NUM_ACTIONS):
            # Predicted next state
            predicted_states = self.B[:, :, a] @ self.beliefs
            predicted_states = np.clip(predicted_states, 1e-16, None)
            predicted_states = predicted_states / predicted_states.sum()

            # Predicted observations
            predicted_obs = self.A @ predicted_states
            predicted_obs = np.clip(predicted_obs, 1e-16, None)
            predicted_obs = predicted_obs / predicted_obs.sum()

            # Pragmatic: E_Q(o')[ln P_C(o')]
            C_shifted = self.C - np.max(self.C)
            log_prefs = C_shifted - np.log(np.sum(np.exp(C_shifted)))
            pragmatic[a] = float(np.sum(predicted_obs * log_prefs))

            # Epistemic: -E_Q(s')[H[P(o'|s')]]
            ent = 0.0
            for s in range(self.A.shape[1]):
                obs_given_s = np.clip(self.A[:, s], 1e-16, None)
                ent -= predicted_states[s] * np.sum(
                    obs_given_s * np.log(obs_given_s)
                )
            epistemic[a] = float(ent)

        total = -self.gamma * pragmatic - epistemic
        self._last_efe_values = total
        self._last_efe_decomposition = {
            'epistemic': epistemic,
            'pragmatic': pragmatic,
            'total': total,
        }
        return self._last_efe_decomposition

    def get_recommended_action(self) -> tuple[int, np.ndarray]:
        """Get the FEP-recommended action via EFE minimization.

        Returns:
            Tuple of (action_idx, action_probs).
        """
        if self._last_efe_values is None:
            self.compute_efe_all_actions()

        return pgmax_bridge._numpy_select_action(
            self.A, self.B, self.C, self.E,
            self.beliefs, self.gamma, self.alpha,
            rng=self.rng,
        )

    # --- Concordia lifecycle hooks ----------------------------------------

    def pre_observe(self, observation: str) -> str:
        """Called when the entity receives an observation.

        Parses the text and updates beliefs. Returns empty string
        (observations are processed internally, not bubbled up).

        Args:
            observation: Text observation from Concordia.

        Returns:
            Empty string (internal processing only).
        """
        self.process_observation(observation)
        return ''

    def pre_act(self) -> str:
        """Called when the entity needs to act.

        Computes EFE, formats FEP context for the LLM prompt.

        Returns:
            FEP assessment string to be included in the LLM prompt.
        """
        return self.get_fep_context()

    def post_act(self, action_attempt: str) -> str:
        """Called after the entity acts.

        Records the action for learning and data accumulation.

        Args:
            action_attempt: The action text from Concordia.

        Returns:
            Empty string.
        """
        # Try to interpret the action
        action_idx = self.bridge.interpret_llm_action(action_attempt)
        action_str = ACTIONS[action_idx]

        # Record in tracker
        if self._last_efe_values is not None:
            self.tracker.record_decision(self._last_efe_values, action_str)

        # Record for adaptive model
        if self.tracker.observation_history:
            self.adaptive_model.record_transition(
                self.tracker.observation_history[-1], action_idx
            )

        return ''

    def update(self) -> None:
        """Called after observation/action cycle.

        Checks if adaptive re-learning should be triggered.
        """
        if self._adaptive and self.adaptive_model.should_relearn(self.tracker):
            self.adaptive_model.relearn()

        # Reset cached EFE
        self._last_efe_values = None
        self._last_efe_decomposition = None

    # --- Context formatting -----------------------------------------------

    def get_fep_context(self) -> str:
        """Generate full FEP context string for LLM prompting.

        Combines belief state, free energy metrics, and drive balance
        into a formatted string.

        Returns:
            Multi-line FEP context string.
        """
        # Compute EFE decomposition
        decomp = self.compute_efe_decomposition()

        # Current VFE
        vfe = self.tracker.mean_recent_vfe(window=3) if self.tracker.vfe_history else 0.0

        # Format beliefs
        beliefs_text = self.bridge.format_beliefs_for_llm(
            self.beliefs, vfe, decomp['total']
        )

        # Format drive balance
        mean_epistemic = float(np.mean(decomp['epistemic']))
        mean_pragmatic = float(np.mean(decomp['pragmatic']))
        drive_text = self.bridge.format_epistemic_drive(
            mean_epistemic, mean_pragmatic
        )

        # Recommended action
        action_idx, action_probs = self.get_recommended_action()
        rec_action = ACTIONS[action_idx]

        # Compose
        lines = [
            beliefs_text,
            '',
            drive_text,
            '',
            f'FEP recommendation: {rec_action} '
            f'(probability: {action_probs[action_idx]:.0%})',
        ]

        # Surprise info
        if self.tracker.surprise_history:
            recent_surprise = self.tracker.mean_recent_surprise(window=3)
            lines.append(
                f'Recent surprise level: {recent_surprise:.2f} nats '
                f'(threshold: {self.tracker.surprise_threshold:.1f})'
            )

        if self.adaptive_model.relearn_count > 0:
            lines.append(
                f'Model has been re-learned {self.adaptive_model.relearn_count} time(s).'
            )

        return '\n'.join(lines)

    def get_state_summary(self) -> dict[str, Any]:
        """Return a summary of the component's internal state.

        Returns:
            Dict with beliefs, free energy, drive balance, and metadata.
        """
        factored = pgmax_bridge.get_factored_beliefs(self.beliefs)

        summary = {
            'name': self.name,
            'role': self.role,
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
            'num_observations': self.tracker.num_observations,
            'num_decisions': self.tracker.num_decisions,
            'relearn_count': self.adaptive_model.relearn_count,
        }

        if self.tracker.surprise_history:
            summary['mean_recent_surprise'] = self.tracker.mean_recent_surprise()
        if self.tracker.vfe_history:
            summary['mean_recent_vfe'] = self.tracker.mean_recent_vfe()

        return summary

    # --- State serialization (Concordia compatibility) --------------------

    def get_state(self) -> dict:
        """Returns serializable state for checkpointing."""
        return {
            'beliefs': self.beliefs.tolist(),
            'E': self.E.tolist(),
            'num_observations': self.tracker.num_observations,
            'num_decisions': self.tracker.num_decisions,
        }

    def set_state(self, state: dict) -> None:
        """Restores state from checkpoint."""
        if 'beliefs' in state:
            self.beliefs = np.array(state['beliefs'])
        if 'E' in state:
            self.E = np.array(state['E'])


# ===========================================================================
# Dashboard: format FEP metrics for display
# ===========================================================================

def format_fep_dashboard_data(
    tracker: FEPBeliefTracker,
    component: Optional[FEPAgentComponent] = None,
) -> dict[str, Any]:
    """Format FEP metrics for dashboard display.

    Produces a structured dict suitable for rendering in a UI or logging.

    Args:
        tracker: The FEPBeliefTracker with recorded history.
        component: Optional FEPAgentComponent for additional context.

    Returns:
        Dict with keys:
        - current_beliefs: dict of state name -> probability
        - free_energy_trajectory: list of VFE values
        - surprise_trajectory: list of surprise values
        - epistemic_pragmatic_balance: dict with drive fractions
        - model_confidence: dict with entropy and confidence metrics
        - action_history: list of actions taken
        - num_observations: total observations
        - num_surprising: count of surprising observations
    """
    dashboard: dict[str, Any] = {}

    # Current beliefs
    if tracker.factored_belief_history:
        last_factored = tracker.factored_belief_history[-1]
        health_beliefs = last_factored[0]
        urgency_beliefs = last_factored[1]
    else:
        health_beliefs = np.ones(NUM_HEALTH) / NUM_HEALTH
        urgency_beliefs = np.ones(NUM_URGENCY) / NUM_URGENCY

    dashboard['current_beliefs'] = {
        'health': {
            PROJECT_HEALTH_STATES[i]: float(health_beliefs[i])
            for i in range(NUM_HEALTH)
        },
        'urgency': {
            TASK_URGENCY_STATES[i]: float(urgency_beliefs[i])
            for i in range(NUM_URGENCY)
        },
    }

    # Free energy trajectory
    dashboard['free_energy_trajectory'] = list(tracker.vfe_history)

    # Surprise trajectory
    dashboard['surprise_trajectory'] = list(tracker.surprise_history)

    # Epistemic vs pragmatic balance
    if component is not None and component._last_efe_decomposition is not None:
        decomp = component._last_efe_decomposition
        mean_epist = float(np.mean(np.abs(decomp['epistemic'])))
        mean_prag = float(np.mean(np.abs(decomp['pragmatic'])))
        total = mean_epist + mean_prag
        if total > 1e-8:
            dashboard['epistemic_pragmatic_balance'] = {
                'epistemic_fraction': mean_epist / total,
                'pragmatic_fraction': mean_prag / total,
                'epistemic_raw': mean_epist,
                'pragmatic_raw': mean_prag,
            }
        else:
            dashboard['epistemic_pragmatic_balance'] = {
                'epistemic_fraction': 0.5,
                'pragmatic_fraction': 0.5,
                'epistemic_raw': 0.0,
                'pragmatic_raw': 0.0,
            }
    else:
        dashboard['epistemic_pragmatic_balance'] = {
            'epistemic_fraction': 0.5,
            'pragmatic_fraction': 0.5,
            'epistemic_raw': 0.0,
            'pragmatic_raw': 0.0,
        }

    # Model confidence metrics
    health_entropy = float(-np.sum(
        health_beliefs * np.log(np.clip(health_beliefs, 1e-16, None))
    ))
    urgency_entropy = float(-np.sum(
        urgency_beliefs * np.log(np.clip(urgency_beliefs, 1e-16, None))
    ))
    max_health_entropy = float(np.log(NUM_HEALTH))
    max_urgency_entropy = float(np.log(NUM_URGENCY))

    dashboard['model_confidence'] = {
        'health_entropy': health_entropy,
        'urgency_entropy': urgency_entropy,
        'health_confidence': 1.0 - (health_entropy / max_health_entropy) if max_health_entropy > 0 else 1.0,
        'urgency_confidence': 1.0 - (urgency_entropy / max_urgency_entropy) if max_urgency_entropy > 0 else 1.0,
    }

    # Action history
    dashboard['action_history'] = list(tracker.action_history)

    # Summary stats
    dashboard['num_observations'] = tracker.num_observations
    dashboard['num_surprising'] = sum(tracker.surprise_flags)
    dashboard['num_decisions'] = tracker.num_decisions

    if component is not None:
        dashboard['relearn_count'] = component.adaptive_model.relearn_count

    return dashboard
