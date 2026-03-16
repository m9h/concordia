"""Tests for the Concordia-FEP integration.

Tests cover:
  1. FEPAgentComponent initialization with SustainHub parameters
  2. Observation encoding/decoding roundtrips via text parsing
  3. Belief updates when observing different health/urgency states
  4. Surprise detection (novel observations flag correctly)
  5. FEP context formatting produces valid LLM prompt text
  6. Adaptive model triggers re-learning after sufficient mismatch
  7. Dashboard data formatting produces expected structure
  8. FEPLLMBridge action interpretation
  9. FEPBeliefTracker history tracking
  10. EFE decomposition into epistemic/pragmatic components

All tests run with the NumPy fallback -- PGMax is NOT required.
"""

import unittest
import numpy as np

from examples.games.sustain_hub import fep_integration
from examples.games.sustain_hub import pgmax_bridge


class TestFEPAgentComponentInit(unittest.TestCase):
    """Test FEPAgentComponent initialization."""

    def test_default_creation(self):
        """Component can be created with default parameters."""
        comp = fep_integration.FEPAgentComponent()
        self.assertEqual(comp.name, 'fep_advisor')
        self.assertEqual(comp.role, 'contributor')
        self.assertEqual(comp.gamma, 1.0)
        self.assertEqual(comp.alpha, 16.0)

    def test_custom_role(self):
        """Component respects custom role parameter."""
        comp = fep_integration.FEPAgentComponent(role='innovator')
        self.assertEqual(comp.role, 'innovator')

    def test_initial_beliefs_match_prior(self):
        """Initial beliefs should equal the D prior from the model."""
        comp = fep_integration.FEPAgentComponent()
        np.testing.assert_array_almost_equal(
            comp.beliefs, comp.adaptive_model.D
        )

    def test_belief_shape(self):
        """Beliefs have correct composite shape (12,)."""
        comp = fep_integration.FEPAgentComponent()
        self.assertEqual(comp.beliefs.shape, (12,))

    def test_sustain_hub_dimensions(self):
        """Component uses correct SustainHub POMDP dimensions."""
        comp = fep_integration.FEPAgentComponent()
        self.assertEqual(comp.A.shape, (9, 12))  # 9 obs, 12 states
        self.assertEqual(comp.B.shape, (12, 12, 6))  # 12 states, 6 actions
        self.assertEqual(comp.C.shape, (9,))
        self.assertEqual(comp.D.shape, (12,))
        self.assertEqual(comp.E.shape, (6,))

    def test_tracker_initialized(self):
        """Tracker is initialized with zero history."""
        comp = fep_integration.FEPAgentComponent()
        self.assertEqual(comp.tracker.num_observations, 0)
        self.assertEqual(comp.tracker.num_decisions, 0)

    def test_adaptive_model_initialized(self):
        """Adaptive model is initialized."""
        comp = fep_integration.FEPAgentComponent()
        self.assertIsNotNone(comp.adaptive_model)
        self.assertEqual(comp.adaptive_model.relearn_count, 0)


class TestObservationEncodingDecoding(unittest.TestCase):
    """Test observation parsing and encoding/decoding roundtrips."""

    def test_text_parsing_high_success(self):
        """Text with 'high harmony' and 'task success' parses correctly."""
        bridge = fep_integration.FEPLLMBridge()
        hi, task = bridge.parse_observation_from_text(
            'The harmony index is high. The task result was a success.'
        )
        self.assertEqual(hi, 'high')
        self.assertEqual(task, 'success')

    def test_text_parsing_low_failure(self):
        """Text with 'low health' and 'task failure' parses correctly."""
        bridge = fep_integration.FEPLLMBridge()
        hi, task = bridge.parse_observation_from_text(
            'The project health is low and declining. The task outcome was a failure.'
        )
        self.assertEqual(hi, 'low')
        self.assertEqual(task, 'failure')

    def test_text_parsing_medium_partial(self):
        """Text with 'medium/moderate' and 'partial' parses correctly."""
        bridge = fep_integration.FEPLLMBridge()
        hi, task = bridge.parse_observation_from_text(
            'The harmony index is moderate. The task result was partial.'
        )
        self.assertEqual(hi, 'medium')
        self.assertEqual(task, 'partial')

    def test_text_parsing_defaults_on_unknown(self):
        """Unknown text defaults to medium/partial."""
        bridge = fep_integration.FEPLLMBridge()
        hi, task = bridge.parse_observation_from_text(
            'The weather is sunny today.'
        )
        self.assertEqual(hi, 'medium')
        self.assertEqual(task, 'partial')

    def test_explicit_observation_overrides_text(self):
        """Explicit hi_level/task_outcome override text parsing."""
        comp = fep_integration.FEPAgentComponent()
        result = comp.process_observation(
            'Some random text',
            hi_level='low',
            task_outcome='failure',
        )
        self.assertEqual(result['hi_level'], 'low')
        self.assertEqual(result['task_outcome'], 'failure')

    def test_observation_index_valid(self):
        """Processed observation yields a valid composite index."""
        comp = fep_integration.FEPAgentComponent()
        result = comp.process_observation(
            '', hi_level='high', task_outcome='success'
        )
        self.assertGreaterEqual(result['obs_idx'], 0)
        self.assertLess(result['obs_idx'], fep_integration.NUM_COMPOSITE_OBS)

    def test_observation_roundtrip_all(self):
        """All HI x task combinations produce valid indices that roundtrip."""
        for hi in fep_integration.HI_OBSERVATIONS:
            for task in fep_integration.TASK_OBSERVATIONS:
                hi_idx = pgmax_bridge.hi_str_to_idx(hi)
                task_idx = pgmax_bridge.task_str_to_idx(task)
                obs_idx = pgmax_bridge.encode_observation(hi_idx, task_idx)
                hi_dec, task_dec = pgmax_bridge.decode_observation(obs_idx)
                self.assertEqual(hi_idx, hi_dec)
                self.assertEqual(task_idx, task_dec)


class TestBeliefUpdates(unittest.TestCase):
    """Test that belief updates respond correctly to observations."""

    def test_beliefs_change_after_observation(self):
        """Beliefs change after processing an observation."""
        comp = fep_integration.FEPAgentComponent()
        initial = comp.beliefs.copy()
        comp.process_observation('', hi_level='low', task_outcome='failure')
        self.assertFalse(
            np.allclose(comp.beliefs, initial),
            'Beliefs should change after observation'
        )

    def test_low_failure_shifts_toward_declining(self):
        """Observing low HI + failure shifts beliefs toward declining."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='low', task_outcome='failure')
        factored = pgmax_bridge.get_factored_beliefs(comp.beliefs)
        health = factored[0]
        # declining (idx 2) should be more probable than healthy (idx 0)
        self.assertGreater(
            health[2], health[0],
            'Low HI + failure should make declining more likely than healthy'
        )

    def test_high_success_shifts_toward_healthy(self):
        """Observing high HI + success shifts beliefs toward healthy."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='high', task_outcome='success')
        factored = pgmax_bridge.get_factored_beliefs(comp.beliefs)
        health = factored[0]
        self.assertEqual(
            np.argmax(health), 0,
            'High HI + success should make healthy most likely'
        )

    def test_repeated_observations_strengthen_belief(self):
        """Repeated consistent observations strengthen the corresponding belief."""
        comp = fep_integration.FEPAgentComponent()
        for _ in range(5):
            comp.process_observation('', hi_level='high', task_outcome='success')
        factored = pgmax_bridge.get_factored_beliefs(comp.beliefs)
        # After 5 consistent positive observations, healthy confidence should be high
        self.assertGreater(factored[0][0], 0.6,
                           'Repeated high/success should yield >60% healthy belief')

    def test_beliefs_sum_to_one(self):
        """Beliefs always sum to 1 after updates."""
        comp = fep_integration.FEPAgentComponent()
        for hi in ['high', 'medium', 'low']:
            for task in ['success', 'partial', 'failure']:
                comp.process_observation('', hi_level=hi, task_outcome=task)
                self.assertAlmostEqual(
                    comp.beliefs.sum(), 1.0, places=6,
                    msg=f'Beliefs do not sum to 1 after ({hi}, {task})'
                )

    def test_different_observations_give_different_beliefs(self):
        """Different observations produce different posterior beliefs."""
        comp1 = fep_integration.FEPAgentComponent(seed=42)
        comp2 = fep_integration.FEPAgentComponent(seed=42)
        comp1.process_observation('', hi_level='high', task_outcome='success')
        comp2.process_observation('', hi_level='low', task_outcome='failure')
        self.assertFalse(
            np.allclose(comp1.beliefs, comp2.beliefs),
            'Different observations should yield different beliefs'
        )


class TestSurpriseDetection(unittest.TestCase):
    """Test that surprise detection works correctly."""

    def test_surprise_recorded(self):
        """Surprise is recorded after each observation."""
        comp = fep_integration.FEPAgentComponent()
        result = comp.process_observation('', hi_level='high', task_outcome='success')
        self.assertIn('surprise', result)
        self.assertIsInstance(result['surprise'], float)
        self.assertEqual(len(comp.tracker.surprise_history), 1)

    def test_surprise_is_finite(self):
        """Surprise values are finite for all observation types."""
        comp = fep_integration.FEPAgentComponent()
        for hi in fep_integration.HI_OBSERVATIONS:
            for task in fep_integration.TASK_OBSERVATIONS:
                result = comp.process_observation('', hi_level=hi, task_outcome=task)
                self.assertTrue(
                    np.isfinite(result['surprise']),
                    f'Surprise not finite for ({hi}, {task}): {result["surprise"]}'
                )

    def test_surprise_flag_for_novel_observation(self):
        """Novel observation after consistent history should flag as surprising."""
        comp = fep_integration.FEPAgentComponent(surprise_threshold=1.5)
        # Build up strong beliefs with consistent observations
        for _ in range(10):
            comp.process_observation('', hi_level='high', task_outcome='success')

        # Now observe something very different
        result = comp.process_observation('', hi_level='low', task_outcome='failure')
        # After seeing many high/success, a low/failure should be more surprising
        # than the consistent observations were
        earlier_surprise = comp.tracker.surprise_history[0]
        novel_surprise = result['surprise']
        self.assertGreater(
            novel_surprise, earlier_surprise,
            'Novel observation after consistent history should be more surprising'
        )

    def test_is_surprising_method(self):
        """tracker.is_surprising() reflects the surprise flag."""
        comp = fep_integration.FEPAgentComponent(surprise_threshold=0.01)
        comp.process_observation('', hi_level='low', task_outcome='failure')
        # With a very low threshold, most observations should be surprising
        self.assertTrue(comp.tracker.is_surprising(lookback=1))

    def test_high_threshold_not_surprising(self):
        """With a very high threshold, observations are not flagged."""
        comp = fep_integration.FEPAgentComponent(surprise_threshold=100.0)
        comp.process_observation('', hi_level='low', task_outcome='failure')
        self.assertFalse(comp.tracker.is_surprising(lookback=1))

    def test_mean_recent_surprise(self):
        """Mean recent surprise is computed correctly."""
        comp = fep_integration.FEPAgentComponent()
        for _ in range(5):
            comp.process_observation('', hi_level='medium', task_outcome='partial')
        mean_s = comp.tracker.mean_recent_surprise(window=3)
        self.assertIsInstance(mean_s, float)
        self.assertTrue(np.isfinite(mean_s))


class TestFEPContextFormatting(unittest.TestCase):
    """Test that FEP context formatting produces valid LLM prompt text."""

    def test_get_fep_context_returns_string(self):
        """get_fep_context() returns a non-empty string."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='medium', task_outcome='partial')
        context = comp.get_fep_context()
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)

    def test_context_contains_key_sections(self):
        """Context string contains belief state, drives, and recommendation."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='high', task_outcome='success')
        context = comp.get_fep_context()
        self.assertIn('FEP Internal State', context)
        self.assertIn('Believed project health:', context)
        self.assertIn('Believed task urgency:', context)
        self.assertIn('FEP recommendation:', context)
        self.assertIn('Motivational balance:', context)

    def test_format_beliefs_for_llm(self):
        """format_beliefs_for_llm produces expected structure."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='low', task_outcome='failure')
        text = fep_integration.FEPLLMBridge.format_beliefs_for_llm(
            comp.beliefs, vfe=0.5, efe_values=comp.compute_efe_all_actions()
        )
        self.assertIn('[FEP Internal State]', text)
        self.assertIn('Model fit (VFE):', text)
        self.assertIn('Expected Free Energy per action', text)

    def test_format_beliefs_includes_all_health_states(self):
        """Formatted beliefs mention all health states."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='medium', task_outcome='partial')
        text = fep_integration.FEPLLMBridge.format_beliefs_for_llm(
            comp.beliefs, vfe=0.3
        )
        for state in fep_integration.PROJECT_HEALTH_STATES:
            self.assertIn(state, text)

    def test_format_epistemic_drive_curiosity(self):
        """Epistemic drive text describes curiosity when epistemic dominates."""
        text = fep_integration.FEPLLMBridge.format_epistemic_drive(
            epistemic_value=-5.0,
            pragmatic_value=-1.0,
        )
        self.assertIn('CURIOSITY', text.upper())

    def test_format_epistemic_drive_exploitation(self):
        """Pragmatic drive text describes exploitation when pragmatic dominates."""
        text = fep_integration.FEPLLMBridge.format_epistemic_drive(
            epistemic_value=-1.0,
            pragmatic_value=-5.0,
        )
        self.assertIn('GOAL-SEEKING', text.upper())

    def test_format_epistemic_drive_balanced(self):
        """Balanced drives produce balanced description."""
        text = fep_integration.FEPLLMBridge.format_epistemic_drive(
            epistemic_value=-3.0,
            pragmatic_value=-3.0,
        )
        self.assertIn('balance', text.lower())

    def test_context_after_multiple_observations(self):
        """Context is well-formed after multiple observations."""
        comp = fep_integration.FEPAgentComponent()
        for hi, task in [('high', 'success'), ('medium', 'partial'), ('low', 'failure')]:
            comp.process_observation('', hi_level=hi, task_outcome=task)
        context = comp.get_fep_context()
        self.assertIn('surprise', context.lower())


class TestAdaptiveModel(unittest.TestCase):
    """Test adaptive model re-learning."""

    def test_should_not_relearn_with_few_data(self):
        """Re-learning should not trigger with too few data points."""
        comp = fep_integration.FEPAgentComponent(min_data_for_relearn=10)
        for _ in range(5):
            comp.process_observation('', hi_level='low', task_outcome='failure')
        self.assertFalse(comp.adaptive_model.should_relearn(comp.tracker))

    def test_should_relearn_with_high_surprise(self):
        """Re-learning triggers when surprise exceeds threshold with enough data."""
        comp = fep_integration.FEPAgentComponent(
            min_data_for_relearn=3,
            relearn_threshold=0.001,  # very low threshold
            surprise_threshold=0.001,
        )
        # Accumulate data and record transitions
        for i in range(5):
            comp.process_observation('', hi_level='low', task_outcome='failure')
            comp.adaptive_model.record_transition(
                comp.tracker.observation_history[-1], i % 6
            )
        # With very low threshold, should trigger
        self.assertTrue(comp.adaptive_model.should_relearn(comp.tracker))

    def test_numpy_relearn_updates_A(self):
        """NumPy fallback re-learning modifies the A matrix."""
        model = fep_integration.AdaptiveGenerativeModel(min_data_for_relearn=2)
        original_A = model.A.copy()
        # Record some transitions
        for i in range(10):
            model.record_transition(i % 9, i % 6)
        success = model._numpy_relearn()
        self.assertTrue(success)
        self.assertFalse(
            np.allclose(model.A, original_A),
            'A matrix should change after re-learning'
        )
        self.assertEqual(model.relearn_count, 1)

    def test_relearn_preserves_column_normalization(self):
        """After re-learning, A matrix columns still sum to 1."""
        model = fep_integration.AdaptiveGenerativeModel(min_data_for_relearn=2)
        for i in range(10):
            model.record_transition(i % 9, i % 6)
        model._numpy_relearn()
        for s in range(fep_integration.NUM_COMPOSITE_STATES):
            col_sum = model.A[:, s].sum()
            self.assertAlmostEqual(
                col_sum, 1.0, places=6,
                msg=f'A column {s} sums to {col_sum} after relearn'
            )

    def test_model_divergence_zero_when_unchanged(self):
        """Model divergence is 0 when A hasn't been modified."""
        model = fep_integration.AdaptiveGenerativeModel()
        div = model.model_divergence()
        self.assertAlmostEqual(div, 0.0, places=10)

    def test_model_divergence_positive_after_relearn(self):
        """Model divergence > 0 after re-learning."""
        model = fep_integration.AdaptiveGenerativeModel(min_data_for_relearn=2)
        for i in range(10):
            model.record_transition(i % 9, i % 6)
        model._numpy_relearn()
        div = model.model_divergence()
        self.assertGreater(div, 0.0)

    def test_reset_to_original(self):
        """reset_to_original() restores A and B matrices."""
        model = fep_integration.AdaptiveGenerativeModel(min_data_for_relearn=2)
        original_A = model.A.copy()
        for i in range(10):
            model.record_transition(i % 9, i % 6)
        model._numpy_relearn()
        model.reset_to_original()
        np.testing.assert_array_almost_equal(model.A, original_A)

    def test_adaptive_relearn_in_update(self):
        """FEPAgentComponent.update() triggers re-learning when conditions met."""
        comp = fep_integration.FEPAgentComponent(
            adaptive=True,
            min_data_for_relearn=3,
            relearn_threshold=0.001,  # very low threshold
        )
        # Accumulate enough data
        for i in range(5):
            comp.process_observation('', hi_level='low', task_outcome='failure')
            comp.adaptive_model.record_transition(
                comp.tracker.observation_history[-1], i % 6
            )
        comp.update()
        # Should have triggered re-learning
        self.assertGreaterEqual(comp.adaptive_model.relearn_count, 1)


class TestFEPLLMBridgeActions(unittest.TestCase):
    """Test FEPLLMBridge action interpretation."""

    def test_exact_match(self):
        """Exact action string matches correctly."""
        for action in fep_integration.ACTIONS:
            idx = fep_integration.FEPLLMBridge.interpret_llm_action(action)
            self.assertEqual(idx, fep_integration.ACTIONS.index(action))

    def test_pattern_match_bug_fix(self):
        """'fix the bug' pattern matches bug_fix."""
        idx = fep_integration.FEPLLMBridge.interpret_llm_action(
            'I should fix the bug in the authentication module'
        )
        self.assertEqual(idx, fep_integration.ACTIONS.index('bug_fix'))

    def test_pattern_match_documentation(self):
        """'write documentation' pattern matches documentation."""
        idx = fep_integration.FEPLLMBridge.interpret_llm_action(
            'Let me write some documentation for the API'
        )
        self.assertEqual(idx, fep_integration.ACTIONS.index('documentation'))

    def test_pattern_match_code_review(self):
        """'review code' pattern matches code_review."""
        idx = fep_integration.FEPLLMBridge.interpret_llm_action(
            'I will review the code in this pull request'
        )
        self.assertEqual(idx, fep_integration.ACTIONS.index('code_review'))

    def test_pattern_match_mentor(self):
        """'mentor the team' pattern matches mentor."""
        idx = fep_integration.FEPLLMBridge.interpret_llm_action(
            'I should mentor the new team member'
        )
        self.assertEqual(idx, fep_integration.ACTIONS.index('mentor'))

    def test_fallback_to_first_available(self):
        """Unrecognized text falls back to first available action."""
        idx = fep_integration.FEPLLMBridge.interpret_llm_action(
            'abcdef xyz 12345'
        )
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, fep_integration.NUM_ACTIONS)

    def test_restricted_available_actions(self):
        """Only returns actions from the available set."""
        idx = fep_integration.FEPLLMBridge.interpret_llm_action(
            'bug_fix',
            available_actions=['documentation', 'mentor'],
        )
        action = fep_integration.ACTIONS[idx]
        self.assertIn(action, ['documentation', 'mentor'])


class TestFEPBeliefTracker(unittest.TestCase):
    """Test FEPBeliefTracker history tracking."""

    def test_empty_tracker(self):
        """Fresh tracker has empty history."""
        tracker = fep_integration.FEPBeliefTracker()
        self.assertEqual(tracker.num_observations, 0)
        self.assertEqual(tracker.num_decisions, 0)
        self.assertFalse(tracker.is_surprising())
        self.assertEqual(tracker.mean_recent_surprise(), 0.0)
        self.assertEqual(tracker.mean_recent_vfe(), 0.0)

    def test_record_observation(self):
        """Recording an observation updates all history lists."""
        tracker = fep_integration.FEPBeliefTracker()
        A = pgmax_bridge.build_flat_A_matrix()
        beliefs = pgmax_bridge.build_flat_D_vector()
        surprise = tracker.record_observation(beliefs, 0, A)
        self.assertEqual(tracker.num_observations, 1)
        self.assertEqual(len(tracker.belief_history), 1)
        self.assertEqual(len(tracker.vfe_history), 1)
        self.assertEqual(len(tracker.surprise_history), 1)
        self.assertIsInstance(surprise, float)
        self.assertTrue(np.isfinite(surprise))

    def test_record_decision(self):
        """Recording a decision updates efe_history and action_history."""
        tracker = fep_integration.FEPBeliefTracker()
        efe = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        tracker.record_decision(efe, 'bug_fix')
        self.assertEqual(tracker.num_decisions, 1)
        self.assertEqual(len(tracker.efe_history), 1)
        self.assertEqual(tracker.action_history[0], 'bug_fix')
        np.testing.assert_array_equal(tracker.efe_history[0], efe)

    def test_factored_belief_history(self):
        """Factored beliefs are stored alongside composite beliefs."""
        tracker = fep_integration.FEPBeliefTracker()
        A = pgmax_bridge.build_flat_A_matrix()
        beliefs = pgmax_bridge.build_flat_D_vector()
        tracker.record_observation(beliefs, 0, A)
        self.assertEqual(len(tracker.factored_belief_history), 1)
        factored = tracker.factored_belief_history[0]
        self.assertEqual(len(factored), 2)  # [health, urgency]
        self.assertEqual(factored[0].shape, (3,))
        self.assertEqual(factored[1].shape, (4,))

    def test_vfe_is_finite(self):
        """VFE values are finite for all observations."""
        tracker = fep_integration.FEPBeliefTracker()
        A = pgmax_bridge.build_flat_A_matrix()
        beliefs = pgmax_bridge.build_flat_D_vector()
        for obs in range(fep_integration.NUM_COMPOSITE_OBS):
            tracker.record_observation(beliefs.copy(), obs, A)
        for vfe in tracker.vfe_history:
            self.assertTrue(np.isfinite(vfe), f'VFE not finite: {vfe}')


class TestEFEDecomposition(unittest.TestCase):
    """Test EFE decomposition into epistemic and pragmatic components."""

    def test_decomposition_keys(self):
        """Decomposition returns epistemic, pragmatic, and total."""
        comp = fep_integration.FEPAgentComponent()
        decomp = comp.compute_efe_decomposition()
        self.assertIn('epistemic', decomp)
        self.assertIn('pragmatic', decomp)
        self.assertIn('total', decomp)

    def test_decomposition_shapes(self):
        """All decomposition arrays have shape (6,)."""
        comp = fep_integration.FEPAgentComponent()
        decomp = comp.compute_efe_decomposition()
        self.assertEqual(decomp['epistemic'].shape, (6,))
        self.assertEqual(decomp['pragmatic'].shape, (6,))
        self.assertEqual(decomp['total'].shape, (6,))

    def test_decomposition_finite(self):
        """All decomposition values are finite."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='medium', task_outcome='partial')
        decomp = comp.compute_efe_decomposition()
        for key in ['epistemic', 'pragmatic', 'total']:
            self.assertTrue(
                np.all(np.isfinite(decomp[key])),
                f'{key} contains non-finite values: {decomp[key]}'
            )

    def test_total_equals_combination(self):
        """Total EFE = -gamma * pragmatic - epistemic."""
        comp = fep_integration.FEPAgentComponent(gamma=1.0)
        decomp = comp.compute_efe_decomposition()
        expected_total = -comp.gamma * decomp['pragmatic'] - decomp['epistemic']
        np.testing.assert_array_almost_equal(
            decomp['total'], expected_total, decimal=6
        )

    def test_efe_matches_bridge_computation(self):
        """EFE from decomposition matches pgmax_bridge._numpy_compute_efe."""
        comp = fep_integration.FEPAgentComponent(gamma=1.0)
        comp.process_observation('', hi_level='medium', task_outcome='partial')
        decomp = comp.compute_efe_decomposition()

        for a in range(fep_integration.NUM_ACTIONS):
            bridge_efe = pgmax_bridge._numpy_compute_efe(
                comp.A, comp.B, comp.C, comp.beliefs, a, comp.gamma
            )
            self.assertAlmostEqual(
                decomp['total'][a], bridge_efe, places=5,
                msg=f'EFE mismatch for action {a}'
            )


class TestDashboardData(unittest.TestCase):
    """Test dashboard data formatting."""

    def test_empty_tracker_dashboard(self):
        """Dashboard works with an empty tracker."""
        tracker = fep_integration.FEPBeliefTracker()
        data = fep_integration.format_fep_dashboard_data(tracker)
        self.assertIsInstance(data, dict)
        self.assertIn('current_beliefs', data)
        self.assertIn('free_energy_trajectory', data)
        self.assertIn('surprise_trajectory', data)
        self.assertIn('epistemic_pragmatic_balance', data)
        self.assertIn('model_confidence', data)
        self.assertIn('action_history', data)
        self.assertIn('num_observations', data)
        self.assertIn('num_surprising', data)
        self.assertIn('num_decisions', data)

    def test_dashboard_after_observations(self):
        """Dashboard contains correct data after observations."""
        comp = fep_integration.FEPAgentComponent()
        for hi, task in [('high', 'success'), ('low', 'failure')]:
            comp.process_observation('', hi_level=hi, task_outcome=task)
        comp.compute_efe_decomposition()  # populate decomposition
        data = fep_integration.format_fep_dashboard_data(comp.tracker, comp)
        self.assertEqual(data['num_observations'], 2)
        self.assertEqual(len(data['free_energy_trajectory']), 2)
        self.assertEqual(len(data['surprise_trajectory']), 2)

    def test_dashboard_beliefs_structure(self):
        """Dashboard beliefs have correct nested structure."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='medium', task_outcome='partial')
        data = fep_integration.format_fep_dashboard_data(comp.tracker, comp)
        beliefs = data['current_beliefs']
        self.assertIn('health', beliefs)
        self.assertIn('urgency', beliefs)
        self.assertEqual(len(beliefs['health']), 3)
        self.assertEqual(len(beliefs['urgency']), 4)

        # Health belief values should sum to ~1
        health_sum = sum(beliefs['health'].values())
        self.assertAlmostEqual(health_sum, 1.0, places=4)

    def test_dashboard_confidence_bounds(self):
        """Model confidence values are in [0, 1]."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='high', task_outcome='success')
        data = fep_integration.format_fep_dashboard_data(comp.tracker, comp)
        conf = data['model_confidence']
        self.assertGreaterEqual(conf['health_confidence'], 0.0)
        self.assertLessEqual(conf['health_confidence'], 1.0)
        self.assertGreaterEqual(conf['urgency_confidence'], 0.0)
        self.assertLessEqual(conf['urgency_confidence'], 1.0)

    def test_dashboard_drive_balance_fractions_sum_to_one(self):
        """Epistemic + pragmatic fractions sum to 1."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='medium', task_outcome='partial')
        comp.compute_efe_decomposition()
        data = fep_integration.format_fep_dashboard_data(comp.tracker, comp)
        balance = data['epistemic_pragmatic_balance']
        total = balance['epistemic_fraction'] + balance['pragmatic_fraction']
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_dashboard_relearn_count(self):
        """Dashboard includes relearn_count when component is provided."""
        comp = fep_integration.FEPAgentComponent()
        data = fep_integration.format_fep_dashboard_data(comp.tracker, comp)
        self.assertIn('relearn_count', data)
        self.assertEqual(data['relearn_count'], 0)


class TestConcordiaLifecycle(unittest.TestCase):
    """Test the Concordia lifecycle hooks (pre_observe, pre_act, post_act, update)."""

    def test_pre_observe_returns_empty(self):
        """pre_observe returns empty string."""
        comp = fep_integration.FEPAgentComponent()
        result = comp.pre_observe('The harmony index is high. Task result: success.')
        self.assertEqual(result, '')

    def test_pre_observe_updates_beliefs(self):
        """pre_observe parses text and updates beliefs."""
        comp = fep_integration.FEPAgentComponent()
        initial = comp.beliefs.copy()
        comp.pre_observe('The harmony index is low. The task outcome was a failure.')
        self.assertFalse(np.allclose(comp.beliefs, initial))

    def test_pre_act_returns_context(self):
        """pre_act returns non-empty FEP context string."""
        comp = fep_integration.FEPAgentComponent()
        comp.pre_observe('The harmony index is medium. Task result: partial.')
        context = comp.pre_act()
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)

    def test_post_act_returns_empty(self):
        """post_act returns empty string."""
        comp = fep_integration.FEPAgentComponent()
        comp.pre_observe('Harmony is medium. Task result: partial.')
        result = comp.post_act('bug_fix')
        self.assertEqual(result, '')

    def test_post_act_records_decision(self):
        """post_act records the action via tracker."""
        comp = fep_integration.FEPAgentComponent()
        comp.pre_observe('Harmony is medium. Task result: partial.')
        # Need EFE computed for recording
        comp.compute_efe_all_actions()
        comp.post_act('bug_fix')
        self.assertEqual(comp.tracker.num_decisions, 1)

    def test_full_lifecycle(self):
        """Complete lifecycle: pre_observe -> pre_act -> post_act -> update."""
        comp = fep_integration.FEPAgentComponent()
        comp.pre_observe('The harmony index is high. Task result: success.')
        context = comp.pre_act()
        self.assertGreater(len(context), 0)
        comp.post_act('bug_fix')
        comp.update()
        # Component should still be in a valid state
        self.assertAlmostEqual(comp.beliefs.sum(), 1.0, places=6)

    def test_multiple_lifecycles(self):
        """Multiple lifecycle iterations maintain valid state."""
        comp = fep_integration.FEPAgentComponent()
        observations = [
            ('The harmony index is high. Task result: success.', 'bug_fix'),
            ('The harmony index is medium. Task result: partial.', 'feature'),
            ('The harmony index is low. Task result: failure.', 'documentation'),
        ]
        for obs_text, action in observations:
            comp.pre_observe(obs_text)
            comp.pre_act()
            comp.post_act(action)
            comp.update()

        self.assertEqual(comp.tracker.num_observations, 3)
        self.assertAlmostEqual(comp.beliefs.sum(), 1.0, places=6)


class TestStateSerialization(unittest.TestCase):
    """Test state get/set for checkpointing."""

    def test_get_state_returns_dict(self):
        """get_state returns a serializable dict."""
        comp = fep_integration.FEPAgentComponent()
        state = comp.get_state()
        self.assertIsInstance(state, dict)
        self.assertIn('beliefs', state)
        self.assertIn('E', state)

    def test_set_state_restores_beliefs(self):
        """set_state restores beliefs from checkpoint."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='low', task_outcome='failure')
        state = comp.get_state()

        # Create new component and restore state
        comp2 = fep_integration.FEPAgentComponent()
        comp2.set_state(state)
        np.testing.assert_array_almost_equal(comp.beliefs, comp2.beliefs)

    def test_get_state_summary(self):
        """get_state_summary returns expected keys."""
        comp = fep_integration.FEPAgentComponent()
        comp.process_observation('', hi_level='medium', task_outcome='partial')
        summary = comp.get_state_summary()
        expected_keys = {
            'name', 'role', 'beliefs_health', 'beliefs_urgency',
            'habits', 'gamma', 'alpha', 'num_observations',
            'num_decisions', 'relearn_count',
        }
        self.assertTrue(expected_keys.issubset(set(summary.keys())))


class TestGetRecommendedAction(unittest.TestCase):
    """Test action recommendation from EFE minimization."""

    def test_returns_valid_action(self):
        """get_recommended_action returns valid index and distribution."""
        comp = fep_integration.FEPAgentComponent(seed=42)
        comp.process_observation('', hi_level='medium', task_outcome='partial')
        action_idx, probs = comp.get_recommended_action()
        self.assertGreaterEqual(action_idx, 0)
        self.assertLess(action_idx, fep_integration.NUM_ACTIONS)
        self.assertAlmostEqual(probs.sum(), 1.0, places=6)
        self.assertTrue(np.all(probs >= 0))

    def test_action_probs_change_with_beliefs(self):
        """Different belief states produce different action recommendations."""
        comp1 = fep_integration.FEPAgentComponent(seed=42)
        comp1.process_observation('', hi_level='high', task_outcome='success')
        _, probs1 = comp1.get_recommended_action()

        comp2 = fep_integration.FEPAgentComponent(seed=42)
        comp2.process_observation('', hi_level='low', task_outcome='failure')
        _, probs2 = comp2.get_recommended_action()

        # Probabilities should differ for different observations
        self.assertFalse(
            np.allclose(probs1, probs2),
            'Action probs should differ for different belief states'
        )


if __name__ == '__main__':
    unittest.main()
