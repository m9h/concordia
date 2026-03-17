"""Unit tests for Phase 1-2 SustainHub community dynamics code.

Tests cover:
  - TrustNetwork: construction, get, update, reciprocity, centralization
  - NormTracker: construction, compliance, emergence, stability
  - BurnoutTracker: construction, delta mechanics, contagion, cascade
  - evaluate.py: _jsd, belief alignment, dialogue acts, coalitions,
    BRS, SUE, CHS
"""

import math

import numpy as np
import pytest

from examples.games.sustain_hub import simulation
from examples.games.sustain_hub import social_data
from examples.games.sustain_hub import evaluate


# =====================================================================
# Helpers
# =====================================================================

def _make_trust_network(
    players=('Alice', 'Bob'),
    initial_trust=0.5,
    lr=0.15,
):
    """Build a small TrustNetwork with uniform initial trust."""
    matrix = {}
    for p in players:
        matrix[p] = {q: initial_trust for q in players if q != p}
    return simulation.TrustNetwork(players, matrix, learning_rate=lr)


def _task_type_map():
    """Canonical task-label -> type mapping for test fixtures."""
    return {
        'Fix: bug A': 'bug_fix',
        'Feature: feat B': 'feature',
        'Docs: doc C': 'documentation',
        'Review: review D': 'code_review',
    }


# =====================================================================
# TrustNetwork tests
# =====================================================================


class TestTrustNetwork:

    def test_construction_from_initial_matrix(self):
        matrix = {
            'Alice': {'Bob': 0.3},
            'Bob': {'Alice': 0.7},
        }
        tn = simulation.TrustNetwork(['Alice', 'Bob'], matrix)
        assert tn.get('Alice', 'Bob') == pytest.approx(0.3)
        assert tn.get('Bob', 'Alice') == pytest.approx(0.7)

    def test_get_returns_correct_values(self):
        tn = _make_trust_network(players=('A', 'B', 'C'), initial_trust=0.2)
        assert tn.get('A', 'B') == pytest.approx(0.2)
        assert tn.get('B', 'C') == pytest.approx(0.2)
        # Self-trust is not stored; should return default 0.0
        assert tn.get('A', 'A') == pytest.approx(0.0)
        # Unknown player returns default 0.0
        assert tn.get('A', 'Z') == pytest.approx(0.0)

    def test_update_complementary_coverage_increases_trust(self):
        """Both agents do non-preferred tasks on different types -> trust up."""
        players = ('Alice', 'Bob')
        tn = _make_trust_network(players, initial_trust=0.0, lr=0.15)
        roles = {
            'Alice': social_data.Role.CONTRIBUTOR,   # preferred = bug_fix
            'Bob': social_data.Role.INNOVATOR,        # preferred = feature
        }
        ttm = _task_type_map()
        # Alice does docs (non-preferred), Bob does review (non-preferred)
        joint = {'Alice': 'Docs: doc C', 'Bob': 'Review: review D'}
        tn.update_after_sprint(joint, ttm, roles, {'Alice': 1.0, 'Bob': 1.0})

        # Signal = +1.0 for complementary non-preferred coverage
        assert tn.get('Alice', 'Bob') > 0.0
        assert tn.get('Bob', 'Alice') > 0.0

    def test_update_free_riding_decreases_trust(self):
        """One agent does preferred while neglected tasks remain -> trust down."""
        players = ('Alice', 'Bob')
        tn = _make_trust_network(players, initial_trust=0.5, lr=0.15)
        roles = {
            'Alice': social_data.Role.CONTRIBUTOR,   # preferred = bug_fix
            'Bob': social_data.Role.INNOVATOR,        # preferred = feature
        }
        ttm = _task_type_map()
        # Alice sacrifices (does docs), Bob free-rides (does feature = preferred)
        # Neglected types remain (code_review, bug_fix not covered)
        joint = {'Alice': 'Docs: doc C', 'Bob': 'Feature: feat B'}
        initial_trust_ab = tn.get('Alice', 'Bob')
        tn.update_after_sprint(joint, ttm, roles, {'Alice': 1.0, 'Bob': 3.0})

        # Alice's trust in Bob should decrease (Bob free-rode while Alice sacrificed)
        assert tn.get('Alice', 'Bob') < initial_trust_ab

    def test_reciprocity_returns_valid_float(self):
        tn = _make_trust_network(
            players=('A', 'B', 'C'), initial_trust=0.5
        )
        r = tn.reciprocity()
        assert isinstance(r, float)
        # With symmetric initial trust, reciprocity should be 1.0
        assert r == pytest.approx(1.0)

    def test_centralization_returns_valid_float(self):
        tn = _make_trust_network(
            players=('A', 'B', 'C'), initial_trust=0.5
        )
        c = tn.centralization()
        assert isinstance(c, float)
        # Uniform trust -> zero centralization
        assert c == pytest.approx(0.0, abs=1e-10)

    def test_single_player_edge_case(self):
        tn = simulation.TrustNetwork(['Solo'], {'Solo': {}})
        assert tn.reciprocity() == pytest.approx(1.0)
        assert tn.centralization() == pytest.approx(0.0)
        assert tn.matrix == {'Solo': {}}

    def test_empty_joint_action(self):
        tn = _make_trust_network(players=('A', 'B'), initial_trust=0.3)
        ttm = _task_type_map()
        roles = {
            'A': social_data.Role.CONTRIBUTOR,
            'B': social_data.Role.INNOVATOR,
        }
        initial = tn.get('A', 'B')
        # Empty joint action -> no tasks addressed -> all types neglected
        tn.update_after_sprint({}, ttm, roles, {})
        # Trust should not crash; values should still be valid
        assert isinstance(tn.get('A', 'B'), float)
        assert -1.0 <= tn.get('A', 'B') <= 1.0

    def test_matrix_property_returns_copy(self):
        tn = _make_trust_network(players=('A', 'B'), initial_trust=0.5)
        m = tn.matrix
        m['A']['B'] = 999.0
        # Original should be unmodified
        assert tn.get('A', 'B') == pytest.approx(0.5)


# =====================================================================
# NormTracker tests
# =====================================================================


class TestNormTracker:

    def test_construction(self):
        nt = simulation.NormTracker(['Alice', 'Bob'])
        norms = nt.norms
        # All norms should start at 0.0
        for name in social_data.EMERGENT_NORMS:
            assert norms[name] == pytest.approx(0.0)
        assert nt.active_norms == {}

    def test_update_after_sprint_returns_compliance_dict(self):
        nt = simulation.NormTracker(['Alice', 'Bob'])
        ttm = _task_type_map()
        roles = {
            'Alice': social_data.Role.CONTRIBUTOR,
            'Bob': social_data.Role.INNOVATOR,
        }
        joint = {'Alice': 'Fix: bug A', 'Bob': 'Feature: feat B'}
        compliance = nt.update_after_sprint(joint, ttm, roles)
        assert isinstance(compliance, dict)
        for norm_name in social_data.EMERGENT_NORMS:
            assert norm_name in compliance
            assert 0.0 <= compliance[norm_name] <= 1.0

    def test_cover_neglected_compliance_when_covering(self):
        """Covering a previously neglected type should give high compliance."""
        nt = simulation.NormTracker(['Alice', 'Bob'])
        ttm = _task_type_map()
        roles = {
            'Alice': social_data.Role.CONTRIBUTOR,
            'Bob': social_data.Role.INNOVATOR,
        }
        # First sprint: only bug_fix and feature covered
        joint1 = {'Alice': 'Fix: bug A', 'Bob': 'Feature: feat B'}
        nt.update_after_sprint(joint1, ttm, roles)

        # Second sprint: cover docs and review (previously neglected)
        joint2 = {'Alice': 'Docs: doc C', 'Bob': 'Review: review D'}
        prev_sprint = {'joint_action': joint1}
        c2 = nt.update_after_sprint(joint2, ttm, roles, prev_sprint=prev_sprint)
        assert c2['cover_neglected'] == pytest.approx(1.0)

    def test_cover_neglected_compliance_when_not_covering(self):
        """Not covering previously neglected types -> lower compliance."""
        nt = simulation.NormTracker(['Alice', 'Bob'])
        ttm = _task_type_map()
        roles = {
            'Alice': social_data.Role.CONTRIBUTOR,
            'Bob': social_data.Role.INNOVATOR,
        }
        # First sprint: only bug_fix and feature covered
        joint1 = {'Alice': 'Fix: bug A', 'Bob': 'Feature: feat B'}
        nt.update_after_sprint(joint1, ttm, roles)

        # Second sprint: again only bug_fix and feature (same types)
        joint2 = {'Alice': 'Fix: bug A', 'Bob': 'Feature: feat B'}
        prev_sprint = {'joint_action': joint1}
        c2 = nt.update_after_sprint(joint2, ttm, roles, prev_sprint=prev_sprint)
        # documentation and code_review were neglected, still not covered
        assert c2['cover_neglected'] == pytest.approx(0.0)

    def test_no_free_riding_compliance_all_preferred(self):
        """When all agents pick preferred and neglected types exist -> low compliance."""
        nt = simulation.NormTracker(['Alice', 'Bob'])
        ttm = _task_type_map()
        roles = {
            'Alice': social_data.Role.CONTRIBUTOR,   # preferred = bug_fix
            'Bob': social_data.Role.INNOVATOR,        # preferred = feature
        }
        # Both pick their preferred, leaving docs and review neglected
        joint = {'Alice': 'Fix: bug A', 'Bob': 'Feature: feat B'}
        c = nt.update_after_sprint(joint, ttm, roles)
        # Both agents are violators for no_free_riding
        assert c['no_free_riding'] == pytest.approx(0.0)

    def test_no_free_riding_compliance_with_mix(self):
        """When agents pick non-preferred and all types covered -> full compliance."""
        nt = simulation.NormTracker(['Alice', 'Bob', 'Carol', 'Dave'])
        ttm = _task_type_map()
        roles = {
            'Alice': social_data.Role.CONTRIBUTOR,       # preferred = bug_fix
            'Bob': social_data.Role.INNOVATOR,            # preferred = feature
            'Carol': social_data.Role.KNOWLEDGE_CURATOR,  # preferred = docs
            'Dave': social_data.Role.MAINTAINER,          # preferred = review
        }
        # Each agent covers a different type, all types covered -> no neglected
        joint = {
            'Alice': 'Fix: bug A',
            'Bob': 'Feature: feat B',
            'Carol': 'Docs: doc C',
            'Dave': 'Review: review D',
        }
        c = nt.update_after_sprint(joint, ttm, roles)
        # All types covered, so no_free_riding compliance = 1.0
        assert c['no_free_riding'] == pytest.approx(1.0)

    def test_norm_emergence_rate_after_multiple_sprints(self):
        """After enough high-compliance sprints, some norms should emerge."""
        players = ['Alice', 'Bob', 'Carol', 'Dave']
        nt = simulation.NormTracker(players)
        ttm = _task_type_map()
        roles = {
            'Alice': social_data.Role.CONTRIBUTOR,
            'Bob': social_data.Role.INNOVATOR,
            'Carol': social_data.Role.KNOWLEDGE_CURATOR,
            'Dave': social_data.Role.MAINTAINER,
        }
        # Alternate full coverage sprints to build compliance
        joint_full = {
            'Alice': 'Fix: bug A',
            'Bob': 'Feature: feat B',
            'Carol': 'Docs: doc C',
            'Dave': 'Review: review D',
        }
        prev = None
        for _ in range(10):
            nt.update_after_sprint(joint_full, ttm, roles, prev_sprint=prev)
            prev = {'joint_action': joint_full}

        rate = nt.norm_emergence_rate()
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0
        # With perfect coverage for 10 sprints, at least some norms emerge
        assert rate > 0.0

    def test_norm_stability_index(self):
        nt = simulation.NormTracker(['Alice', 'Bob'])
        # No history -> stability = 1.0
        assert nt.norm_stability_index() == pytest.approx(1.0)

    def test_mean_compliance_rate(self):
        nt = simulation.NormTracker(['Alice', 'Bob'])
        # No sprints -> compliance = 1.0 (default)
        assert nt.mean_compliance_rate() == pytest.approx(1.0)

        ttm = _task_type_map()
        roles = {
            'Alice': social_data.Role.CONTRIBUTOR,
            'Bob': social_data.Role.INNOVATOR,
        }
        joint = {'Alice': 'Fix: bug A', 'Bob': 'Feature: feat B'}
        nt.update_after_sprint(joint, ttm, roles)
        rate = nt.mean_compliance_rate()
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0


# =====================================================================
# BurnoutTracker tests
# =====================================================================


class TestBurnoutTracker:

    def test_construction_starts_at_zero(self):
        bt = simulation.BurnoutTracker(['Alice', 'Bob'])
        for v in bt.levels.values():
            assert v == pytest.approx(0.0)

    def test_burnout_increases_with_nonpreferred_task(self):
        bt = simulation.BurnoutTracker(['Alice'])
        ttm = _task_type_map()
        roles = {'Alice': social_data.Role.CONTRIBUTOR}  # preferred = bug_fix
        # Alice does docs (non-preferred)
        joint = {'Alice': 'Docs: doc C'}
        bt.update_after_sprint(joint, ttm, roles, {'Alice': 1.0}, hi=0.5)
        assert bt.levels['Alice'] > 0.0

    def test_burnout_increases_with_task_failure(self):
        bt = simulation.BurnoutTracker(['Alice'])
        ttm = _task_type_map()
        roles = {'Alice': social_data.Role.CONTRIBUTOR}
        # Alice does preferred task but fails (negative score)
        joint = {'Alice': 'Fix: bug A'}
        bt.update_after_sprint(joint, ttm, roles, {'Alice': -1.0}, hi=0.5)
        assert bt.levels['Alice'] > 0.0

    def test_burnout_decreases_with_preferred_success_and_high_hi(self):
        bt = simulation.BurnoutTracker(['Alice'])
        ttm = _task_type_map()
        roles = {'Alice': social_data.Role.CONTRIBUTOR}  # preferred = bug_fix

        # First pump up burnout with non-preferred + failure
        joint_bad = {'Alice': 'Docs: doc C'}
        for _ in range(5):
            bt.update_after_sprint(
                joint_bad, ttm, roles, {'Alice': -1.0}, hi=0.3
            )
        burnout_after_stress = bt.levels['Alice']
        assert burnout_after_stress > 0.0

        # Now do preferred task, positive score, high HI
        joint_good = {'Alice': 'Fix: bug A'}
        bt.update_after_sprint(
            joint_good, ttm, roles, {'Alice': 3.0}, hi=0.9
        )
        # Burnout should decrease: -0.15 (preferred success) - 0.05 (hi>0.7) = -0.20
        assert bt.levels['Alice'] < burnout_after_stress

    def test_contagion_through_trust_network(self):
        """Burned-out agent with high trust should spread burnout via contagion."""
        players = ('Alice', 'Bob')
        bt = simulation.BurnoutTracker(players)
        tn = _make_trust_network(players, initial_trust=0.9, lr=0.15)
        ttm = _task_type_map()
        roles = {
            'Alice': social_data.Role.CONTRIBUTOR,
            'Bob': social_data.Role.INNOVATOR,
        }

        # Push Alice's burnout above 0.6 threshold
        joint_bad = {'Alice': 'Docs: doc C', 'Bob': 'Feature: feat B'}
        for _ in range(8):
            bt.update_after_sprint(
                joint_bad, ttm, roles, {'Alice': -1.0, 'Bob': 3.0}, hi=0.3,
                trust_network=tn,
            )

        # Alice should be significantly burned out
        assert bt.levels['Alice'] > 0.6

        # Bob should have received some contagion (his burnout > 0)
        assert bt.levels['Bob'] > 0.0

    def test_cascade_count(self):
        bt = simulation.BurnoutTracker(['Alice', 'Bob'])
        # No sprints -> no cascades
        assert bt.cascade_count() == 0

    def test_mean_burnout(self):
        bt = simulation.BurnoutTracker(['Alice', 'Bob'])
        assert bt.mean_burnout() == pytest.approx(0.0)

    def test_peak_burnout(self):
        bt = simulation.BurnoutTracker(['Alice'])
        ttm = _task_type_map()
        roles = {'Alice': social_data.Role.CONTRIBUTOR}

        # Pump up burnout
        for _ in range(3):
            bt.update_after_sprint(
                {'Alice': 'Docs: doc C'}, ttm, roles,
                {'Alice': -1.0}, hi=0.3,
            )
        peak = bt.peak_burnout()
        assert peak > 0.0
        assert peak == max(
            max(snap.values()) for snap in bt.history
        )


# =====================================================================
# evaluate.py: _jsd tests
# =====================================================================


class TestJSD:

    def test_symmetry(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.6, 0.3])
        assert evaluate._jsd(p, q) == pytest.approx(evaluate._jsd(q, p), abs=1e-12)

    def test_bounded_by_ln2(self):
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        jsd = evaluate._jsd(p, q)
        assert jsd >= 0.0
        assert jsd <= math.log(2) + 1e-10

    def test_identical_distributions_zero(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert evaluate._jsd(p, p) == pytest.approx(0.0, abs=1e-10)


# =====================================================================
# evaluate.py: compute_belief_alignment
# =====================================================================


class TestComputeBeliefAlignment:

    def test_empty_snapshots(self):
        result = evaluate.compute_belief_alignment([])
        assert result['bai_per_sprint'] == []
        assert result['mean_bai'] == pytest.approx(1.0)

    def test_single_agent(self):
        sprint = {
            'belief_snapshots': {
                'Alice': {'beliefs_health': {'code_quality': 0.8, 'docs': 0.5}}
            }
        }
        result = evaluate.compute_belief_alignment([sprint])
        # < 2 agents -> BAI = 1.0
        assert result['bai_per_sprint'] == [1.0]
        assert result['mean_bai'] == pytest.approx(1.0)

    def test_two_aligned_agents(self):
        beliefs = {'code_quality': 0.8, 'docs': 0.5, 'infra': 0.3}
        sprint = {
            'belief_snapshots': {
                'Alice': {'beliefs_health': dict(beliefs)},
                'Bob': {'beliefs_health': dict(beliefs)},
            }
        }
        result = evaluate.compute_belief_alignment([sprint])
        assert result['bai_per_sprint'][0] == pytest.approx(1.0, abs=1e-6)

    def test_two_divergent_agents(self):
        sprint = {
            'belief_snapshots': {
                'Alice': {'beliefs_health': {'a': 0.9, 'b': 0.1}},
                'Bob': {'beliefs_health': {'a': 0.1, 'b': 0.9}},
            }
        }
        result = evaluate.compute_belief_alignment([sprint])
        # Beliefs are very divergent -> BAI should be noticeably below 1.0
        assert result['bai_per_sprint'][0] < 0.8


# =====================================================================
# evaluate.py: classify_dialogue_acts
# =====================================================================


class TestClassifyDialogueActs:

    def test_empty_input(self):
        result = evaluate.classify_dialogue_acts([])
        assert result['act_counts'] == {cat: 0 for cat in evaluate.DIALOGUE_ACT_CATEGORIES}
        assert result['coordination_index'] == pytest.approx(0.0)

    def test_text_with_coordination_keywords(self):
        narrative = [
            {
                'Alice': {
                    'Strategy': "Let's divide the work and coordinate between us."
                }
            }
        ]
        result = evaluate.classify_dialogue_acts(narrative)
        assert result['act_counts']['coordinate'] >= 1
        assert result['coordination_index'] > 0.0

    def test_social_pressure_index_with_criticism(self):
        narrative = [
            {
                'Alice': {
                    'Strategy': 'Bob is not pulling weight and is free riding.'
                },
                'Bob': {
                    'Strategy': 'I appreciate the great job everyone is doing.'
                },
            }
        ]
        result = evaluate.classify_dialogue_acts(narrative)
        # Alice has 'criticize' counts, Bob has 'encourage' counts
        assert result['act_counts']['criticize'] >= 1
        assert result['act_counts']['encourage'] >= 1
        assert isinstance(result['social_pressure_index'], float)


# =====================================================================
# evaluate.py: detect_coalitions
# =====================================================================


class TestDetectCoalitions:

    def test_empty_history(self):
        result = evaluate.detect_coalitions([])
        assert result['coalitions_per_sprint'] == []
        assert result['stable_coalitions'] == []
        assert result['num_coalitions_per_sprint'] == []

    def test_all_agents_aligned_one_coalition(self):
        beliefs = {'a': 0.5, 'b': 0.3, 'c': 0.2}
        sprint = {
            'belief_snapshots': {
                'Alice': {'beliefs_health': dict(beliefs)},
                'Bob': {'beliefs_health': dict(beliefs)},
                'Carol': {'beliefs_health': dict(beliefs)},
            }
        }
        result = evaluate.detect_coalitions([sprint])
        # All identical beliefs -> 1 coalition
        assert result['num_coalitions_per_sprint'] == [1]
        assert len(result['coalitions_per_sprint'][0]) == 1
        assert sorted(result['coalitions_per_sprint'][0][0]) == ['Alice', 'Bob', 'Carol']

    def test_agents_divergent_multiple_coalitions(self):
        sprint = {
            'belief_snapshots': {
                'Alice': {'beliefs_health': {'a': 0.99, 'b': 0.01}},
                'Bob': {'beliefs_health': {'a': 0.01, 'b': 0.99}},
            }
        }
        result = evaluate.detect_coalitions([sprint])
        # Very divergent beliefs -> separate coalitions
        assert result['num_coalitions_per_sprint'][0] == 2


# =====================================================================
# evaluate.py: compute_brs
# =====================================================================


class TestComputeBRS:

    def test_always_preferred_brs_zero(self):
        """Agent always does preferred task -> BRS = 0."""
        sprint_history = [
            {'joint_action': {'Alice': 'Fix: bug A'}},
            {'joint_action': {'Alice': 'Fix: bug A'}},
            {'joint_action': {'Alice': 'Fix: bug A'}},
        ]
        player_roles = {'Alice': 'Contributor'}
        brs = evaluate.compute_brs(sprint_history, player_roles)
        assert brs['Alice'] == pytest.approx(0.0)

    def test_always_off_preferred(self):
        """Agent never does preferred task -> BRS = 1.0."""
        sprint_history = [
            {'joint_action': {'Alice': 'Feature: feat B'}},
            {'joint_action': {'Alice': 'Feature: feat B'}},
            {'joint_action': {'Alice': 'Feature: feat B'}},
        ]
        player_roles = {'Alice': 'Contributor'}  # preferred = bug_fix
        brs = evaluate.compute_brs(sprint_history, player_roles)
        # max_consecutive=3, len(tasks)=3 -> BRS = 1.0
        assert brs['Alice'] == pytest.approx(1.0)


# =====================================================================
# evaluate.py: compute_sue
# =====================================================================


class TestComputeSUE:

    def test_all_preferred_tasks(self):
        sprint_history = [
            {
                'joint_action': {
                    'Alice': 'Fix: bug A',
                    'Bob': 'Feature: feat B',
                }
            },
        ]
        player_roles = {
            'Alice': 'Contributor',   # preferred = bug_fix
            'Bob': 'Innovator',       # preferred = feature
        }
        sue = evaluate.compute_sue(sprint_history, player_roles)
        assert sue == pytest.approx(1.0)

    def test_all_unrelated_tasks(self):
        sprint_history = [
            {
                'joint_action': {
                    'Alice': 'Feature: feat B',   # Contributor -> pref bug_fix
                    'Bob': 'Docs: doc C',          # Innovator -> pref feature
                }
            },
        ]
        player_roles = {
            'Alice': 'Contributor',
            'Bob': 'Innovator',
        }
        sue = evaluate.compute_sue(sprint_history, player_roles)
        # Alice: feature is adjacent to bug_fix -> 0.5
        # Bob: documentation is not adjacent to feature -> 0.0
        # Mean = 0.25
        assert sue == pytest.approx(0.25)


# =====================================================================
# evaluate.py: compute_chs
# =====================================================================


class TestComputeCHS:

    def test_basic_weighted_sum(self):
        hi = 0.8
        mean_brs = 0.2
        sue = 0.9
        rq = 0.7
        # Default weights (0.25, 0.25, 0.25, 0.25)
        expected = 0.25 * hi + 0.25 * (1.0 - mean_brs) + 0.25 * sue + 0.25 * rq
        chs = evaluate.compute_chs(hi, mean_brs, sue, rq)
        assert chs == pytest.approx(expected)

    def test_custom_weights(self):
        chs = evaluate.compute_chs(1.0, 0.0, 1.0, 1.0, weights=(0.5, 0.2, 0.2, 0.1))
        expected = 0.5 * 1.0 + 0.2 * 1.0 + 0.2 * 1.0 + 0.1 * 1.0
        assert chs == pytest.approx(expected)

    def test_perfect_scores(self):
        chs = evaluate.compute_chs(1.0, 0.0, 1.0, 1.0)
        assert chs == pytest.approx(1.0)

    def test_worst_scores(self):
        chs = evaluate.compute_chs(0.0, 1.0, 0.0, 0.0)
        assert chs == pytest.approx(0.0)
