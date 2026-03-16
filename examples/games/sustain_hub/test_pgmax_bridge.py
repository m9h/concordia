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

"""Tests for the PGMax Active Inference bridge for SustainHub.

Tests cover:
  1. Generative model construction and matrix shapes
  2. Observation encoding/decoding roundtrips
  3. Action encoding/decoding roundtrips
  4. State encoding/decoding roundtrips
  5. Matrix normalization (columns sum to 1)
  6. Agent observation-action loop
  7. Belief updating changes beliefs
  8. Habit learning modifies E-vector
  9. Agent state summary format
  10. Compatibility with the hand-rolled ActiveInferenceAgent interface

All tests run with the NumPy fallback, so PGMax is NOT required.
If PGMax is available, the tests verify the PGMax backend is activated.
"""

import unittest
import numpy as np

from examples.games.sustain_hub import pgmax_bridge
from examples.games.sustain_hub import active_inference as aif


class TestEncodingDecoding(unittest.TestCase):
    """Test observation, state, and action encoding/decoding."""

    def test_state_encoding_roundtrip(self):
        """Every (health, urgency) pair encodes and decodes correctly."""
        for h in range(pgmax_bridge.NUM_HEALTH):
            for u in range(pgmax_bridge.NUM_URGENCY):
                composite = pgmax_bridge.encode_state(h, u)
                h_dec, u_dec = pgmax_bridge.decode_state(composite)
                self.assertEqual(h, h_dec, f"health mismatch at ({h},{u})")
                self.assertEqual(u, u_dec, f"urgency mismatch at ({h},{u})")

    def test_state_encoding_range(self):
        """All composite state indices are in [0, NUM_COMPOSITE_STATES)."""
        indices = set()
        for h in range(pgmax_bridge.NUM_HEALTH):
            for u in range(pgmax_bridge.NUM_URGENCY):
                idx = pgmax_bridge.encode_state(h, u)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, pgmax_bridge.NUM_COMPOSITE_STATES)
                indices.add(idx)
        # All indices should be unique and cover the full range
        self.assertEqual(len(indices), pgmax_bridge.NUM_COMPOSITE_STATES)

    def test_observation_encoding_roundtrip(self):
        """Every (hi, task) pair encodes and decodes correctly."""
        for hi in range(pgmax_bridge.NUM_HI_OBS):
            for task in range(pgmax_bridge.NUM_TASK_OBS):
                composite = pgmax_bridge.encode_observation(hi, task)
                hi_dec, task_dec = pgmax_bridge.decode_observation(composite)
                self.assertEqual(hi, hi_dec)
                self.assertEqual(task, task_dec)

    def test_observation_encoding_range(self):
        """All composite observation indices are in [0, NUM_COMPOSITE_OBS)."""
        indices = set()
        for hi in range(pgmax_bridge.NUM_HI_OBS):
            for task in range(pgmax_bridge.NUM_TASK_OBS):
                idx = pgmax_bridge.encode_observation(hi, task)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, pgmax_bridge.NUM_COMPOSITE_OBS)
                indices.add(idx)
        self.assertEqual(len(indices), pgmax_bridge.NUM_COMPOSITE_OBS)

    def test_action_str_to_idx(self):
        """All SustainHub actions map to valid integer indices."""
        for i, action in enumerate(pgmax_bridge.ACTIONS):
            self.assertEqual(pgmax_bridge.action_str_to_idx(action), i)

    def test_action_idx_to_str(self):
        """All integer action indices map back to correct strings."""
        for i in range(pgmax_bridge.NUM_ACTIONS):
            action = pgmax_bridge.action_idx_to_str(i)
            self.assertEqual(action, pgmax_bridge.ACTIONS[i])

    def test_action_roundtrip(self):
        """String -> index -> string is identity."""
        for action in pgmax_bridge.ACTIONS:
            idx = pgmax_bridge.action_str_to_idx(action)
            self.assertEqual(pgmax_bridge.action_idx_to_str(idx), action)

    def test_hi_str_to_idx(self):
        """HI level strings map correctly."""
        self.assertEqual(pgmax_bridge.hi_str_to_idx('high'), 0)
        self.assertEqual(pgmax_bridge.hi_str_to_idx('medium'), 1)
        self.assertEqual(pgmax_bridge.hi_str_to_idx('low'), 2)

    def test_task_str_to_idx(self):
        """Task outcome strings map correctly."""
        self.assertEqual(pgmax_bridge.task_str_to_idx('success'), 0)
        self.assertEqual(pgmax_bridge.task_str_to_idx('partial'), 1)
        self.assertEqual(pgmax_bridge.task_str_to_idx('failure'), 2)

    def test_discretize_hi(self):
        """Continuous HI values discretize correctly."""
        self.assertEqual(pgmax_bridge.discretize_hi(0.9), 'high')
        self.assertEqual(pgmax_bridge.discretize_hi(0.7), 'high')
        self.assertEqual(pgmax_bridge.discretize_hi(0.5), 'medium')
        self.assertEqual(pgmax_bridge.discretize_hi(0.4), 'medium')
        self.assertEqual(pgmax_bridge.discretize_hi(0.3), 'low')
        self.assertEqual(pgmax_bridge.discretize_hi(0.0), 'low')

    def test_discretize_task_outcome(self):
        """Reward values discretize correctly."""
        self.assertEqual(pgmax_bridge.discretize_task_outcome(3.0), 'success')
        self.assertEqual(pgmax_bridge.discretize_task_outcome(2.0), 'success')
        self.assertEqual(pgmax_bridge.discretize_task_outcome(1.0), 'partial')
        self.assertEqual(pgmax_bridge.discretize_task_outcome(0.0), 'partial')
        self.assertEqual(pgmax_bridge.discretize_task_outcome(-1.0), 'failure')


class TestGenerativeModel(unittest.TestCase):
    """Test SustainHubGenerativeModel construction."""

    def setUp(self):
        self.model = pgmax_bridge.SustainHubGenerativeModel(
            role='contributor',
            health_prior='uncertain',
        )

    def test_dimensions(self):
        """Model has correct composite dimensions."""
        self.assertEqual(self.model.num_states, 12)
        self.assertEqual(self.model.num_obs, 9)
        self.assertEqual(self.model.num_actions, 6)

    def test_A_matrix_shape(self):
        """A matrix has shape (num_obs, num_states)."""
        self.assertEqual(self.model.A.shape, (9, 12))

    def test_B_matrix_shape(self):
        """B matrix has shape (num_states, num_states, num_actions)."""
        self.assertEqual(self.model.B.shape, (12, 12, 6))

    def test_C_vector_shape(self):
        """C vector has shape (num_obs,)."""
        self.assertEqual(self.model.C.shape, (9,))

    def test_D_vector_shape(self):
        """D vector has shape (num_states,)."""
        self.assertEqual(self.model.D.shape, (12,))

    def test_E_vector_shape(self):
        """E vector has shape (num_actions,)."""
        self.assertEqual(self.model.E.shape, (6,))

    def test_A_columns_sum_to_one(self):
        """Each column of A (per state) sums to 1 (valid distribution)."""
        for s in range(self.model.num_states):
            col_sum = self.model.A[:, s].sum()
            self.assertAlmostEqual(col_sum, 1.0, places=6,
                                   msg=f"A column {s} sums to {col_sum}")

    def test_B_columns_sum_to_one(self):
        """Each column of B (per state, per action) sums to 1."""
        for a in range(self.model.num_actions):
            for s in range(self.model.num_states):
                col_sum = self.model.B[:, s, a].sum()
                self.assertAlmostEqual(col_sum, 1.0, places=6,
                                       msg=f"B column (s={s}, a={a}) sums to {col_sum}")

    def test_D_sums_to_one(self):
        """D vector sums to 1 (valid distribution)."""
        self.assertAlmostEqual(self.model.D.sum(), 1.0, places=6)

    def test_E_sums_to_one(self):
        """E vector sums to 1 (valid distribution)."""
        self.assertAlmostEqual(self.model.E.sum(), 1.0, places=6)

    def test_A_nonnegative(self):
        """A matrix is nonnegative."""
        self.assertTrue(np.all(self.model.A >= 0))

    def test_B_nonnegative(self):
        """B matrix is nonnegative."""
        self.assertTrue(np.all(self.model.B >= 0))

    def test_D_nonnegative(self):
        """D vector is nonnegative."""
        self.assertTrue(np.all(self.model.D >= 0))

    def test_E_nonnegative(self):
        """E vector is nonnegative."""
        self.assertTrue(np.all(self.model.E >= 0))

    def test_different_roles_give_different_E(self):
        """Different roles produce different habit priors."""
        contributor_model = pgmax_bridge.SustainHubGenerativeModel(role='contributor')
        innovator_model = pgmax_bridge.SustainHubGenerativeModel(role='innovator')
        self.assertFalse(np.allclose(contributor_model.E, innovator_model.E),
                         "Contributor and innovator should have different E vectors")

    def test_different_health_priors_give_different_D(self):
        """Different health priors produce different D vectors."""
        uncertain = pgmax_bridge.SustainHubGenerativeModel(health_prior='uncertain')
        optimistic = pgmax_bridge.SustainHubGenerativeModel(health_prior='optimistic')
        self.assertFalse(np.allclose(uncertain.D, optimistic.D),
                         "Uncertain and optimistic should have different D vectors")

    def test_factored_beliefs_marginalization(self):
        """Factored beliefs marginalize correctly from composite beliefs."""
        # Construct beliefs concentrated on state 0 (healthy + balanced)
        beliefs = np.zeros(12)
        beliefs[0] = 1.0
        factored = self.model.get_factored_beliefs(beliefs)
        # Health should be [1, 0, 0], urgency should be [1, 0, 0, 0]
        np.testing.assert_array_almost_equal(factored[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(factored[1], [1.0, 0.0, 0.0, 0.0])

    def test_factored_beliefs_uniform(self):
        """Uniform composite beliefs marginalize to uniform factored beliefs."""
        beliefs = np.ones(12) / 12.0
        factored = self.model.get_factored_beliefs(beliefs)
        np.testing.assert_array_almost_equal(
            factored[0], np.ones(3) / 3.0, decimal=6
        )
        np.testing.assert_array_almost_equal(
            factored[1], np.ones(4) / 4.0, decimal=6
        )


class TestPGMaxAgent(unittest.TestCase):
    """Test the PGMaxAgent agent loop."""

    def setUp(self):
        np.random.seed(42)
        self.agent = pgmax_bridge.PGMaxAgent(
            name='TestAgent',
            role='contributor',
            gamma=1.0,
            alpha=16.0,
            learning_rate=0.1,
            health_prior='uncertain',
        )

    def test_agent_creation(self):
        """Agent can be created with default parameters."""
        self.assertEqual(self.agent.name, 'TestAgent')
        self.assertEqual(self.agent.role, 'contributor')
        self.assertEqual(len(self.agent.beliefs), 12)

    def test_initial_beliefs_match_prior(self):
        """Initial beliefs should equal the D prior."""
        np.testing.assert_array_almost_equal(
            self.agent.beliefs, self.agent.gen_model.D
        )

    def test_observe_changes_beliefs(self):
        """Observing changes the agent's beliefs."""
        initial_beliefs = self.agent.beliefs.copy()
        self.agent.observe('low', 'failure')
        self.assertFalse(np.allclose(self.agent.beliefs, initial_beliefs),
                         "Beliefs should change after observation")

    def test_observe_low_failure_shifts_toward_declining(self):
        """Observing low HI + failure should shift beliefs toward declining states."""
        self.agent.observe('low', 'failure')
        factored = self.agent.get_factored_beliefs()
        # Declining (index 2) should be more probable than healthy (index 0)
        self.assertGreater(factored[0][2], factored[0][0],
                           "Low HI + failure should make declining more likely than healthy")

    def test_observe_high_success_shifts_toward_healthy(self):
        """Observing high HI + success should shift beliefs toward healthy states."""
        self.agent.observe('high', 'success')
        factored = self.agent.get_factored_beliefs()
        # Healthy (index 0) should be most probable
        self.assertEqual(np.argmax(factored[0]), 0,
                         "High HI + success should make healthy most likely")

    def test_decide_returns_valid_action(self):
        """decide() returns a valid action string and probability distribution."""
        action, probs = self.agent.decide()
        self.assertIn(action, pgmax_bridge.ACTIONS)
        self.assertEqual(len(probs), pgmax_bridge.NUM_ACTIONS)
        self.assertAlmostEqual(probs.sum(), 1.0, places=6)
        self.assertTrue(np.all(probs >= 0))

    def test_decide_records_history(self):
        """decide() appends to action_history."""
        self.assertEqual(len(self.agent.action_history), 0)
        self.agent.decide()
        self.assertEqual(len(self.agent.action_history), 1)
        self.agent.decide()
        self.assertEqual(len(self.agent.action_history), 2)

    def test_learn_changes_habits(self):
        """learn() modifies the E-vector."""
        self.agent.decide()  # need an action in history first
        initial_E = self.agent.E.copy()
        self.agent.learn(1.0)  # strong positive outcome
        self.assertFalse(np.allclose(self.agent.E, initial_E),
                         "E-vector should change after learning")

    def test_learn_positive_increases_last_action(self):
        """Positive outcome increases the habit weight for the last action."""
        self.agent.decide()
        last_action = self.agent.action_history[-1]
        last_idx = pgmax_bridge.action_str_to_idx(last_action)
        old_weight = self.agent.E[last_idx]
        self.agent.learn(1.0)
        new_weight = self.agent.E[last_idx]
        # Due to normalization, we check that the weight did not drop
        self.assertGreaterEqual(new_weight, old_weight - 0.01,
                                "Positive learning should not decrease action weight significantly")

    def test_full_sprint_loop(self):
        """Test a complete observe-decide-learn loop (one sprint)."""
        # Sprint 1: observe, decide, learn
        self.agent.observe('medium', 'success')
        action, probs = self.agent.decide()
        self.agent.learn(0.5)

        # Sprint 2: observe, decide, learn
        self.agent.observe('low', 'partial')
        action2, probs2 = self.agent.decide()
        self.agent.learn(-0.5)

        # Agent should have 2 actions in history
        self.assertEqual(len(self.agent.action_history), 2)
        self.assertEqual(len(self.agent.belief_history), 2)

    def test_multiple_sprints_beliefs_evolve(self):
        """Over multiple sprints, beliefs should evolve."""
        beliefs_over_time = [self.agent.beliefs.copy()]

        for obs_hi, obs_task in [('high', 'success'), ('medium', 'partial'),
                                  ('low', 'failure'), ('medium', 'success')]:
            self.agent.observe(obs_hi, obs_task)
            self.agent.decide()
            self.agent.learn(0.0)
            beliefs_over_time.append(self.agent.beliefs.copy())

        # Not all belief vectors should be identical
        all_same = all(np.allclose(beliefs_over_time[0], b)
                       for b in beliefs_over_time[1:])
        self.assertFalse(all_same, "Beliefs should evolve over sprints")

    def test_get_state_summary_format(self):
        """get_state_summary() returns the expected keys."""
        summary = self.agent.get_state_summary()
        expected_keys = {
            'name', 'role', 'backend', 'beliefs_health', 'beliefs_urgency',
            'habits', 'gamma', 'alpha', 'num_composite_states', 'num_composite_obs',
        }
        self.assertEqual(set(summary.keys()), expected_keys)

        # Check nested structure
        self.assertEqual(len(summary['beliefs_health']), 3)
        self.assertEqual(len(summary['beliefs_urgency']), 4)
        self.assertEqual(len(summary['habits']), 6)

        # Values should be valid probabilities (approximately sum to 1)
        health_sum = sum(summary['beliefs_health'].values())
        self.assertAlmostEqual(health_sum, 1.0, places=4)
        urgency_sum = sum(summary['beliefs_urgency'].values())
        self.assertAlmostEqual(urgency_sum, 1.0, places=4)

    def test_state_summary_compatible_with_hand_rolled(self):
        """State summary has the same keys as ActiveInferenceAgent.get_state_summary()."""
        hand_rolled = aif.ActiveInferenceAgent(
            name='TestHandRolled', role='contributor'
        )
        hr_summary = hand_rolled.get_state_summary()
        pg_summary = self.agent.get_state_summary()

        # Core keys that must match for compatibility
        for key in ['name', 'role', 'beliefs_health', 'beliefs_urgency',
                    'habits', 'gamma', 'alpha']:
            self.assertIn(key, pg_summary,
                          f"PGMaxAgent summary missing key '{key}'")

        # Same health state labels
        self.assertEqual(
            set(hr_summary['beliefs_health'].keys()),
            set(pg_summary['beliefs_health'].keys()),
        )
        # Same urgency state labels
        self.assertEqual(
            set(hr_summary['beliefs_urgency'].keys()),
            set(pg_summary['beliefs_urgency'].keys()),
        )
        # Same action labels
        self.assertEqual(
            set(hr_summary['habits'].keys()),
            set(pg_summary['habits'].keys()),
        )


class TestFormatAIFContext(unittest.TestCase):
    """Test the LLM context formatter."""

    def test_format_returns_string(self):
        """format_aif_context_for_llm produces a non-empty string."""
        agent = pgmax_bridge.PGMaxAgent(name='Alice', role='maintainer')
        context = pgmax_bridge.format_aif_context_for_llm(agent)
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)

    def test_format_contains_key_info(self):
        """Context string contains health, urgency, and action info."""
        agent = pgmax_bridge.PGMaxAgent(name='Bob', role='innovator')
        context = pgmax_bridge.format_aif_context_for_llm(agent)
        self.assertIn('Project health belief:', context)
        self.assertIn('Current urgency:', context)
        self.assertIn('Learned task preferences:', context)
        self.assertIn('[Internal Assessment', context)


class TestNumpyFallback(unittest.TestCase):
    """Test that the NumPy fallback produces sensible results."""

    def test_numpy_update_beliefs_normalization(self):
        """Beliefs after update sum to 1."""
        A = pgmax_bridge.build_flat_A_matrix()
        D = pgmax_bridge.build_flat_D_vector()
        for obs_idx in range(pgmax_bridge.NUM_COMPOSITE_OBS):
            updated = pgmax_bridge._numpy_update_beliefs(A, D.copy(), obs_idx)
            self.assertAlmostEqual(updated.sum(), 1.0, places=6)

    def test_numpy_efe_finite(self):
        """EFE computation returns finite values for all actions."""
        model = pgmax_bridge.SustainHubGenerativeModel(role='contributor')
        beliefs = model.D.copy()
        for a in range(model.num_actions):
            efe = pgmax_bridge._numpy_compute_efe(
                model.A, model.B, model.C, beliefs, a
            )
            self.assertTrue(np.isfinite(efe),
                            f"EFE for action {a} is not finite: {efe}")

    def test_numpy_select_action_valid(self):
        """Action selection returns valid index and distribution."""
        model = pgmax_bridge.SustainHubGenerativeModel(role='contributor')
        beliefs = model.D.copy()
        np.random.seed(0)
        idx, probs = pgmax_bridge._numpy_select_action(
            model.A, model.B, model.C, model.E, beliefs
        )
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, model.num_actions)
        self.assertAlmostEqual(probs.sum(), 1.0, places=6)
        self.assertTrue(np.all(probs >= 0))


class TestFlattenFactorBeliefs(unittest.TestCase):
    """Test flatten/unflatten roundtrip for factor beliefs."""

    def test_roundtrip(self):
        """Flatten then unflatten recovers original factor beliefs."""
        health = np.array([0.6, 0.3, 0.1])
        urgency = np.array([0.4, 0.3, 0.2, 0.1])
        joint = pgmax_bridge.flatten_factor_beliefs(health, urgency)
        self.assertEqual(joint.shape, (pgmax_bridge.NUM_COMPOSITE_STATES,))
        self.assertAlmostEqual(joint.sum(), 1.0, places=6)
        recovered = pgmax_bridge.get_factored_beliefs(joint)
        np.testing.assert_array_almost_equal(recovered[0], health)
        np.testing.assert_array_almost_equal(recovered[1], urgency)

    def test_uniform_roundtrip(self):
        """Uniform factors produce uniform joint, which roundtrips."""
        health = np.ones(3) / 3.0
        urgency = np.ones(4) / 4.0
        joint = pgmax_bridge.flatten_factor_beliefs(health, urgency)
        expected = np.ones(12) / 12.0
        np.testing.assert_array_almost_equal(joint, expected)
        recovered = pgmax_bridge.get_factored_beliefs(joint)
        np.testing.assert_array_almost_equal(recovered[0], health)
        np.testing.assert_array_almost_equal(recovered[1], urgency)

    def test_peaked_health(self):
        """Peaked health + uniform urgency roundtrips."""
        health = np.array([1.0, 0.0, 0.0])
        urgency = np.ones(4) / 4.0
        joint = pgmax_bridge.flatten_factor_beliefs(health, urgency)
        recovered = pgmax_bridge.get_factored_beliefs(joint)
        np.testing.assert_array_almost_equal(recovered[0], health)
        np.testing.assert_array_almost_equal(recovered[1], urgency)


class TestPGMaxAvailability(unittest.TestCase):
    """Test PGMax availability detection."""

    def test_pgmax_available_returns_bool(self):
        """pgmax_available() returns a boolean."""
        result = pgmax_bridge.pgmax_available()
        self.assertIsInstance(result, bool)

    def test_agent_reports_backend(self):
        """Agent correctly reports which backend it uses."""
        agent = pgmax_bridge.PGMaxAgent(name='Test', role='contributor')
        summary = agent.get_state_summary()
        self.assertIn(summary['backend'], ('pgmax', 'numpy_fallback'))


class TestPGMaxIntegration(unittest.TestCase):
    """Tests that exercise the actual PGMax backend, skipped if unavailable."""

    def setUp(self):
        if not pgmax_bridge.pgmax_available():
            self.skipTest('PGMax not available; skipping integration tests')

    def test_pgmax_agent_uses_pgmax_backend(self):
        """When PGMax is available, agent should use it."""
        agent = pgmax_bridge.PGMaxAgent(name='PGTest', role='contributor')
        self.assertTrue(agent.uses_pgmax)

    def test_pgmax_model_dimensions(self):
        """PGMax GenerativeModel has correct dimensions."""
        gm = pgmax_bridge.SustainHubGenerativeModel()
        model = gm.pgmax_model
        self.assertIsNotNone(model)
        self.assertEqual(model.num_modalities, 1)
        self.assertEqual(model.num_factors, 1)
        self.assertEqual(model.num_states, [12])
        self.assertEqual(model.num_obs, [9])
        self.assertEqual(model.num_actions, [6])
        # T=1, single-factor: num_policies == num_actions == 6
        self.assertEqual(model.num_policies, 6)
        self.assertEqual(model.policies.shape, (6, 1, 1))

    def test_pgmax_observe_decide_cycle(self):
        """PGMax agent can observe and decide."""
        agent = pgmax_bridge.PGMaxAgent(
            name='PGTest', role='contributor', seed=42
        )
        agent.observe('high', 'success')
        action, probs = agent.decide()
        self.assertIn(action, pgmax_bridge.ACTIONS)
        self.assertAlmostEqual(probs.sum(), 1.0, places=4)


class TestCrossValidation(unittest.TestCase):
    """Cross-validate PGMaxAgent against hand-rolled ActiveInferenceAgent.

    The flattened single-factor approximation won't produce identical
    results to the factored 2-factor model, but the qualitative behavior
    should match: same observation should shift beliefs in the same direction.
    """

    def test_both_agents_agree_on_belief_direction_low_failure(self):
        """Both agents shift toward declining after observing low+failure."""
        np.random.seed(42)
        hr = aif.ActiveInferenceAgent(name='HR', role='contributor')
        pg = pgmax_bridge.PGMaxAgent(name='PG', role='contributor')

        # Both observe the same bad signal
        hr.observe('low', 'failure')
        pg.observe('low', 'failure')

        hr_health = hr.beliefs[0]  # [healthy, stressed, declining]
        pg_factored = pg.get_factored_beliefs()
        pg_health = pg_factored[0]  # [healthy, stressed, declining]

        # Both should have declining as most probable (or at least > healthy)
        self.assertGreater(hr_health[2], hr_health[0],
                           "Hand-rolled: declining should exceed healthy")
        self.assertGreater(pg_health[2], pg_health[0],
                           "PGMax bridge: declining should exceed healthy")

    def test_both_agents_agree_on_belief_direction_high_success(self):
        """Both agents shift toward healthy after observing high+success."""
        np.random.seed(42)
        hr = aif.ActiveInferenceAgent(name='HR', role='contributor')
        pg = pgmax_bridge.PGMaxAgent(name='PG', role='contributor')

        hr.observe('high', 'success')
        pg.observe('high', 'success')

        hr_health = hr.beliefs[0]
        pg_health = pg.get_factored_beliefs()[0]

        # Both should favor healthy
        self.assertEqual(np.argmax(hr_health), 0,
                         "Hand-rolled: healthy should be most likely")
        self.assertEqual(np.argmax(pg_health), 0,
                         "PGMax bridge: healthy should be most likely")


if __name__ == '__main__':
    unittest.main()
