import unittest
import numpy as np

from examples.games.sustain_hub import active_inference as aif

class TestActiveInference(unittest.TestCase):

    def test_agent_creation(self):
        agent = aif.ActiveInferenceAgent(
            name="Alice",
            role="bug_fix",
            gamma=1.0,
            alpha=16.0,
            learning_rate=0.1,
            health_prior="uncertain"
        )
        state_summary = agent.get_state_summary()

        # Verify health beliefs
        health_beliefs = state_summary['beliefs_health']
        self.assertAlmostEqual(sum(health_beliefs.values()), 1.0)
        for val in health_beliefs.values():
            self.assertGreaterEqual(val, 0.0)

        # Verify urgency beliefs
        urgency_beliefs = state_summary['beliefs_urgency']
        self.assertAlmostEqual(sum(urgency_beliefs.values()), 1.0)
        for val in urgency_beliefs.values():
            self.assertGreaterEqual(val, 0.0)

    def test_observe_updates_beliefs(self):
        agent = aif.ActiveInferenceAgent(
            name="Alice",
            role="bug_fix",
            gamma=1.0,
            alpha=16.0,
            learning_rate=0.1,
            health_prior="uncertain"
        )

        initial_state = agent.get_state_summary()
        prior_declining = initial_state['beliefs_health']['declining']

        # Observe "low" harmony index and "failure" task outcome
        # These correspond to negative indicators, which should increase
        # the belief in 'declining' health.
        agent.observe("low", "failure")

        post_state = agent.get_state_summary()
        post_declining = post_state['beliefs_health']['declining']

        self.assertGreater(post_declining, prior_declining)

    def test_decide_returns_valid_action(self):
        agent = aif.ActiveInferenceAgent(
            name="Alice",
            role="bug_fix",
            gamma=1.0,
            alpha=16.0,
            learning_rate=0.1,
            health_prior="uncertain"
        )

        action, probs = agent.decide()

        self.assertIn(action, aif.ACTIONS)
        self.assertEqual(len(probs), len(aif.ACTIONS))
        self.assertAlmostEqual(np.sum(probs), 1.0)

    def test_learn_updates_habits(self):
        agent = aif.ActiveInferenceAgent(
            name="Alice",
            role="bug_fix",
            gamma=1.0,
            alpha=16.0,
            learning_rate=0.1,
            health_prior="uncertain"
        )

        # Decide to create an action history
        action, _ = agent.decide()

        # Get prior habit value
        prior_habit = agent.get_state_summary()['habits'][action]

        # Learn from a positive valence outcome
        agent.learn(1.0)

        # Get updated habit value
        post_habit = agent.get_state_summary()['habits'][action]

        # The probability (habit) of the chosen action should increase
        # after a positive outcome.
        self.assertGreater(post_habit, prior_habit)

    def test_softmax_basic(self):
        # Uniform distribution
        probs_uniform = aif._softmax(np.array([0, 0, 0]))
        np.testing.assert_allclose(probs_uniform, [1/3, 1/3, 1/3])

        # Sharp distribution
        probs_sharp = aif._softmax(np.array([100, 0, 0]))
        np.testing.assert_allclose(probs_sharp, [1.0, 0.0, 0.0], atol=1e-10)

    def test_A_matrix_shape(self):
        A = aif.build_A_matrix()

        self.assertEqual(len(A), 2)

        # A{1}: Harmony Index observation
        self.assertEqual(
            A[0].shape,
            (aif.NUM_HI_OBS, aif.NUM_HEALTH, aif.NUM_URGENCY)
        )

        # A{2}: Task outcome
        self.assertEqual(
            A[1].shape,
            (aif.NUM_TASK_OBS, aif.NUM_HEALTH, aif.NUM_URGENCY)
        )

    def test_B_matrix_shape(self):
        B = aif.build_B_matrix()

        self.assertEqual(len(B), 2)

        # B{1}: Health transitions
        self.assertEqual(
            B[0].shape,
            (aif.NUM_HEALTH, aif.NUM_HEALTH, aif.NUM_ACTIONS)
        )

        # B{2}: Urgency transitions
        self.assertEqual(
            B[1].shape,
            (aif.NUM_URGENCY, aif.NUM_URGENCY, aif.NUM_ACTIONS)
        )

    def test_different_roles_different_preferences(self):
        agent_contributor = aif.ActiveInferenceAgent(
            name="Alice",
            role="contributor",
            gamma=1.0,
            alpha=16.0,
            learning_rate=0.1,
            health_prior="uncertain"
        )

        agent_innovator = aif.ActiveInferenceAgent(
            name="Bob",
            role="innovator",
            gamma=1.0,
            alpha=16.0,
            learning_rate=0.1,
            health_prior="uncertain"
        )

        habits_contributor = agent_contributor.get_state_summary()['habits']
        habits_innovator = agent_innovator.get_state_summary()['habits']

        # Their habit priors (E-vector) should not be identical
        self.assertNotEqual(habits_contributor, habits_innovator)

        # Specifically, the contributor should have higher preference for bug_fix
        self.assertGreater(
            habits_contributor['bug_fix'],
            habits_innovator['bug_fix']
        )

        # And the innovator should have higher preference for feature
        self.assertGreater(
            habits_innovator['feature'],
            habits_contributor['feature']
        )

if __name__ == '__main__':
    unittest.main()
