"""Unit tests for SustainHubPayoff: dropout enforcement and RQ calculation."""

import random
from examples.games.sustain_hub import simulation
from examples.games.sustain_hub import social_data


def make_payoff(
    num_sprints=3,
    stress_schedule=None,
    dropout_name=None,
):
  """Create a minimal SustainHubPayoff for testing."""
  names = ["Alice", "Bob", "Carol", "Dave"]
  roles = {
      "Alice": social_data.Role.CONTRIBUTOR,
      "Bob": social_data.Role.INNOVATOR,
      "Carol": social_data.Role.KNOWLEDGE_CURATOR,
      "Dave": social_data.Role.MAINTAINER,
  }
  tasks = ["Fix login bug", "Add dark mode", "Update README", "Review PR #42"]
  task_type_map = {
      "Fix login bug": "bug_fix",
      "Add dark mode": "feature",
      "Update README": "documentation",
      "Review PR #42": "code_review",
  }
  rel_matrix = {n: {m: 0.5 for m in names if m != n} for n in names}

  return simulation.SustainHubPayoff(
      player_names=names,
      player_roles=roles,
      task_options=tasks,
      task_type_map=task_type_map,
      relational_matrix=rel_matrix,
      num_sprints=num_sprints,
      stress_schedule=stress_schedule,
      dropout_name=dropout_name,
  )


def test_basic_scoring():
  """Verify basic scoring works without stress."""
  payoff = make_payoff(num_sprints=3)
  random.seed(42)

  joint = {
      "Alice": "Fix login bug",   # preferred
      "Bob": "Add dark mode",     # preferred
      "Carol": "Update README",   # preferred
      "Dave": "Review PR #42",    # preferred
  }
  scores = payoff.action_to_scores(joint)
  assert len(scores) == 4, f"Expected 4 scores, got {len(scores)}"
  print(f"Sprint 1 scores: {scores}")
  print(f"Sprint 1 HI: {payoff.harmony_index():.3f}")
  print("PASS: basic_scoring")


def test_dropout_enforcement():
  """Verify dropout agent is forced to skip on the correct sprint."""
  stress_schedule = {2: "contributor_dropout", 3: "task_overload"}
  payoff = make_payoff(
      num_sprints=3,
      stress_schedule=stress_schedule,
      dropout_name="Carol",
  )
  random.seed(42)

  # Sprint 1: no stress — all agents act normally
  joint1 = {
      "Alice": "Fix login bug",
      "Bob": "Add dark mode",
      "Carol": "Update README",
      "Dave": "Review PR #42",
  }
  scores1 = payoff.action_to_scores(joint1)
  recorded1 = payoff.sprint_history[-1]
  assert recorded1["stress"] is None, f"Sprint 1 stress should be None, got {recorded1['stress']}"
  assert recorded1["joint_action"]["Carol"] == "Update README", \
      f"Sprint 1: Carol should have real action, got {recorded1['joint_action']['Carol']}"
  print(f"Sprint 1: Carol action = {recorded1['joint_action']['Carol']}, stress = {recorded1['stress']}")

  # Sprint 2: contributor_dropout — Carol should be forced to skip
  # Even if Carol's action is provided (e.g. from partial joint action with None
  # being replaced by Concordia), the enforcement should override it.
  joint2 = {
      "Alice": "Fix login bug",
      "Bob": "Add dark mode",
      "Carol": None,  # Not a participant — would be None from PayoffMatrix
      "Dave": "Review PR #42",
  }
  scores2 = payoff.action_to_scores(joint2)
  recorded2 = payoff.sprint_history[-1]
  assert recorded2["stress"] == "contributor_dropout", \
      f"Sprint 2 stress should be contributor_dropout, got {recorded2['stress']}"
  assert recorded2["joint_action"]["Carol"] == "Skip this sprint", \
      f"Sprint 2: Carol should be 'Skip this sprint', got {recorded2['joint_action']['Carol']}"
  assert scores2["Carol"] == social_data.REWARD_SKIP, \
      f"Sprint 2: Carol score should be {social_data.REWARD_SKIP}, got {scores2['Carol']}"
  print(f"Sprint 2: Carol action = {recorded2['joint_action']['Carol']}, stress = {recorded2['stress']}")

  # Sprint 3: task_overload — Carol should be back
  joint3 = {
      "Alice": "Fix login bug",
      "Bob": "Add dark mode",
      "Carol": "Update README",
      "Dave": "Review PR #42",
  }
  scores3 = payoff.action_to_scores(joint3)
  recorded3 = payoff.sprint_history[-1]
  assert recorded3["stress"] == "task_overload", \
      f"Sprint 3 stress should be task_overload, got {recorded3['stress']}"
  assert recorded3["joint_action"]["Carol"] == "Update README", \
      f"Sprint 3: Carol should have real action, got {recorded3['joint_action']['Carol']}"
  print(f"Sprint 3: Carol action = {recorded3['joint_action']['Carol']}, stress = {recorded3['stress']}")

  print("PASS: dropout_enforcement")


def test_rq_calculation():
  """Verify RQ < 1.0 when stress causes degradation."""
  stress_schedule = {2: "contributor_dropout", 3: "task_overload"}
  payoff = make_payoff(
      num_sprints=3,
      stress_schedule=stress_schedule,
      dropout_name="Carol",
  )
  random.seed(42)

  # Sprint 1: all succeed with preferred tasks (high HI)
  joint1 = {
      "Alice": "Fix login bug",
      "Bob": "Add dark mode",
      "Carol": "Update README",
      "Dave": "Review PR #42",
  }
  payoff.action_to_scores(joint1)
  hi1 = payoff.sprint_history[-1]["harmony_index"]

  # Sprint 2: Carol drops out (lower HI expected)
  joint2 = {
      "Alice": "Fix login bug",
      "Bob": "Add dark mode",
      "Carol": None,
      "Dave": "Review PR #42",
  }
  payoff.action_to_scores(joint2)
  hi2 = payoff.sprint_history[-1]["harmony_index"]

  # Sprint 3: task overload
  joint3 = {
      "Alice": "Fix login bug",
      "Bob": "Add dark mode",
      "Carol": "Update README",
      "Dave": "Review PR #42",
  }
  payoff.action_to_scores(joint3)
  hi3 = payoff.sprint_history[-1]["harmony_index"]

  rq = payoff.resilience_quotient
  print(f"HI trajectory: [{hi1:.3f}, {hi2:.3f}, {hi3:.3f}]")
  print(f"Pre-stress HI (Sprint 1): {hi1:.3f}")
  print(f"Stress HI (Sprints 2-3): {(hi2 + hi3) / 2:.3f}")
  print(f"RQ = {rq:.3f}")

  # RQ should differ from 1.0 when stress affects HI
  assert rq != 1.0, f"RQ should not be exactly 1.0 with stress, got {rq}"
  assert len(payoff.sprint_history) == 3, f"Expected 3 sprints, got {len(payoff.sprint_history)}"
  assert payoff.sprint_history[0]["stress"] is None
  assert payoff.sprint_history[1]["stress"] == "contributor_dropout"
  assert payoff.sprint_history[2]["stress"] == "task_overload"
  print(f"RQ != 1.0: {rq != 1.0} (rq={rq:.3f})")
  print("PASS: rq_calculation")


def test_policy_vote_does_not_increment_sprint():
  """Verify that a policy vote does NOT append to sprint_history."""
  payoff = make_payoff(num_sprints=3)

  # A policy vote
  vote = {
      "Alice": "Continue as-is",
      "Bob": "Implement Tool Subsidy",
      "Carol": "Continue as-is",
      "Dave": "Implement Maintenance Premium",
  }
  scores = payoff.action_to_scores(vote)
  assert len(payoff.sprint_history) == 0, \
      f"Policy vote should NOT add to sprint_history, got {len(payoff.sprint_history)}"
  assert all(s == 0.0 for s in scores.values()), "Policy vote scores should all be 0"
  print("PASS: policy_vote_does_not_increment_sprint")


if __name__ == "__main__":
  test_basic_scoring()
  print()
  test_dropout_enforcement()
  print()
  test_rq_calculation()
  print()
  test_policy_vote_does_not_increment_sprint()
  print()
  print("=" * 40)
  print("ALL TESTS PASSED")
