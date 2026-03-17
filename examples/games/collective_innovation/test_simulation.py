"""Unit tests for Collective Innovation simulation components.

Tests cover RecipeBook, InventoryState, ConnectivityManager, and
InnovationPayoff -- the core, LLM-free building blocks of the simulation.
"""

import math
import os
import random

import pytest

from examples.games.collective_innovation.simulation import (
    ConnectivityManager,
    InnovationPayoff,
    InventoryState,
    RecipeBook,
)
from examples.games.collective_innovation import social_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recipe_book() -> RecipeBook:
    """Load the recipe book once for the whole module."""
    return RecipeBook()


@pytest.fixture
def agents() -> list[str]:
    return ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]


# ===========================================================================
# RecipeBook tests
# ===========================================================================


class TestRecipeBook:

    def test_loads_without_error(self, recipe_book: RecipeBook):
        """alchemy_data.json loads without raising."""
        assert recipe_book is not None

    def test_num_entities_positive(self, recipe_book: RecipeBook):
        assert recipe_book.num_entities > 0

    def test_num_recipes_positive(self, recipe_book: RecipeBook):
        assert recipe_book.num_recipes > 0

    def test_try_combine_known_recipe_water_fire(self, recipe_book: RecipeBook):
        """water + fire -> steam (per alchemy_data.json)."""
        result = recipe_book.try_combine("water", "fire")
        assert result == "steam"

    def test_try_combine_known_recipe_fire_earth(self, recipe_book: RecipeBook):
        """fire + earth -> lava."""
        result = recipe_book.try_combine("fire", "earth")
        assert result == "lava"

    def test_try_combine_known_recipe_water_earth(self, recipe_book: RecipeBook):
        """water + earth -> mud."""
        result = recipe_book.try_combine("water", "earth")
        assert result == "mud"

    def test_try_combine_unknown_recipe_returns_none(
        self, recipe_book: RecipeBook
    ):
        """Nonsense element names should return None."""
        assert recipe_book.try_combine("xyzzy", "plugh") is None
        assert recipe_book.try_combine("banana", "spaceship") is None

    def test_try_combine_order_independent(self, recipe_book: RecipeBook):
        """a+b should equal b+a."""
        assert recipe_book.try_combine("water", "fire") == recipe_book.try_combine("fire", "water")
        assert recipe_book.try_combine("fire", "earth") == recipe_book.try_combine("earth", "fire")

    def test_try_combine_case_insensitive(self, recipe_book: RecipeBook):
        """Water + Fire == water + fire."""
        assert recipe_book.try_combine("Water", "Fire") == recipe_book.try_combine("water", "fire")
        assert recipe_book.try_combine("EARTH", "FIRE") == recipe_book.try_combine("earth", "fire")

    def test_get_all_recipes_returns_tuples(self, recipe_book: RecipeBook):
        recipes = recipe_book.get_all_recipes()
        assert len(recipes) > 0
        # Each recipe is (parent1, parent2, child)
        for r in recipes[:10]:
            assert len(r) == 3

    def test_get_possible_discoveries(self, recipe_book: RecipeBook):
        """With all four base elements, at least some discoveries are possible."""
        base = {"water", "fire", "earth", "air"}
        possible = recipe_book.get_possible_discoveries(base)
        assert len(possible) > 0
        # steam, lava, mud should all be reachable from base elements
        assert "steam" in possible
        assert "lava" in possible
        assert "mud" in possible


# ===========================================================================
# InventoryState tests
# ===========================================================================


class TestInventoryState:

    def test_starts_with_initial_elements(self):
        inv = InventoryState(elements=set(social_data.STARTING_ITEMS))
        assert inv.elements == {"water", "fire", "earth", "air"}

    def test_adding_element_increases_count(self):
        inv = InventoryState(elements=set(social_data.STARTING_ITEMS))
        assert len(inv.elements) == 4
        inv.elements.add("steam")
        assert len(inv.elements) == 5
        assert "steam" in inv.elements

    def test_elements_property_returns_set(self):
        inv = InventoryState(elements=set(social_data.STARTING_ITEMS))
        assert isinstance(inv.elements, set)

    def test_discovery_history_starts_empty(self):
        inv = InventoryState(elements=set(social_data.STARTING_ITEMS))
        assert inv.discovery_history == []

    def test_failed_attempts_starts_empty(self):
        inv = InventoryState(elements=set(social_data.STARTING_ITEMS))
        assert inv.failed_attempts == []

    def test_default_factory_creates_empty(self):
        inv = InventoryState()
        assert inv.elements == set()
        assert inv.discovery_history == []
        assert inv.failed_attempts == []


# ===========================================================================
# ConnectivityManager tests
# ===========================================================================


class TestConnectivityManager:

    def test_fully_connected_all_in_one_group(self, agents: list[str]):
        cm = ConnectivityManager(agents, mode="fully_connected")
        groups = cm.get_groups(step=0)
        assert len(groups) == 1
        assert sorted(groups[0]) == sorted(agents)

    def test_isolated_each_agent_alone(self, agents: list[str]):
        cm = ConnectivityManager(agents, mode="isolated")
        groups = cm.get_groups(step=0)
        assert len(groups) == len(agents)
        for group in groups:
            assert len(group) == 1
        # All agents represented
        all_members = [m for g in groups for m in g]
        assert sorted(all_members) == sorted(agents)

    def test_dynamic_agents_in_subgroups(self, agents: list[str]):
        cm = ConnectivityManager(
            agents, mode="dynamic", group_size=4, visit_prob=0.0
        )
        groups = cm.get_groups(step=0)
        # With 8 agents and group_size=4, should have 2 groups
        assert len(groups) == 2
        # Each group should have 4 agents (no visits when prob=0)
        for group in groups:
            assert len(group) == 4

    def test_dynamic_wrap_around(self):
        """First and last base groups are adjacent in dynamic mode.

        With 3 groups (indices 0, 1, 2), group 0 should be adjacent to
        group 2 via wrap-around, and group 2 adjacent to group 0.
        """
        # 12 agents, group_size=4 -> 3 groups (indices 0, 1, 2)
        agent_names = [f"Agent_{i}" for i in range(12)]
        rng = random.Random(42)
        cm = ConnectivityManager(
            agent_names, mode="dynamic", group_size=4,
            visit_prob=1.0,  # force visits
            visit_duration=100,
            rng=rng,
        )

        # Run a few steps to trigger visits
        for step in range(5):
            cm.get_groups(step=step)

        # Check that at least one agent from group 0 visited group 2
        # or vice versa (wrap-around adjacency).
        # The adjacency options for group 0 are [2, 1] and for group 2 are [1, 0].
        visits = cm._active_visits
        group_0_agents = set(agent_names[:4])
        group_2_agents = set(agent_names[8:12])

        wrap_around_visit_found = False
        for agent, (dest, _) in visits.items():
            if agent in group_0_agents and dest == 2:
                wrap_around_visit_found = True
                break
            if agent in group_2_agents and dest == 0:
                wrap_around_visit_found = True
                break

        assert wrap_around_visit_found, (
            "Expected at least one wrap-around visit between group 0 and "
            "group 2 over 5 steps with visit_prob=1.0"
        )

    def test_get_groups_returns_correct_structure(self, agents: list[str]):
        """get_groups returns list[list[str]]."""
        for mode in ["fully_connected", "isolated", "dynamic"]:
            cm = ConnectivityManager(agents, mode=mode, group_size=4)
            groups = cm.get_groups(step=0)
            assert isinstance(groups, list)
            for group in groups:
                assert isinstance(group, list)
                for member in group:
                    assert isinstance(member, str)

    def test_dynamic_no_agent_lost(self, agents: list[str]):
        """All agents appear in exactly one group after dynamic shuffling."""
        rng = random.Random(123)
        cm = ConnectivityManager(
            agents, mode="dynamic", group_size=4,
            visit_prob=0.5, visit_duration=3, rng=rng,
        )
        for step in range(10):
            groups = cm.get_groups(step=step)
            all_members = [m for g in groups for m in g]
            assert sorted(all_members) == sorted(agents), (
                f"Step {step}: agents were lost or duplicated"
            )

    def test_unknown_mode_raises(self, agents: list[str]):
        cm = ConnectivityManager(agents, mode="unknown")
        with pytest.raises(ValueError, match="Unknown connectivity mode"):
            cm.get_groups(step=0)


# ===========================================================================
# InnovationPayoff tests
# ===========================================================================


class TestInnovationPayoff:

    def _make_payoff(
        self,
        recipe_book: RecipeBook,
        player_names: list[str] | None = None,
        mode: str = "fully_connected",
    ) -> InnovationPayoff:
        if player_names is None:
            player_names = ["Alice", "Bob"]
        cm = ConnectivityManager(player_names, mode=mode)
        return InnovationPayoff(
            player_names=player_names,
            recipe_book=recipe_book,
            connectivity_manager=cm,
        )

    # -- _parse_combination ---------------------------------------------------

    def test_parse_combination_comma_separated(self, recipe_book: RecipeBook):
        payoff = self._make_payoff(recipe_book)
        e1, e2 = payoff._parse_combination("water, fire")
        assert e1 == "water"
        assert e2 == "fire"

    def test_parse_combination_with_prefix(self, recipe_book: RecipeBook):
        payoff = self._make_payoff(recipe_book)
        e1, e2 = payoff._parse_combination("I combine water and fire")
        assert e1 is not None and e2 is not None
        assert set([e1.lower(), e2.lower()]) == {"water", "fire"}

    def test_parse_combination_garbage_returns_none(
        self, recipe_book: RecipeBook
    ):
        payoff = self._make_payoff(recipe_book)
        e1, e2 = payoff._parse_combination("askdjhaskdjh")
        assert e1 is None
        assert e2 is None

    def test_parse_combination_empty_string(self, recipe_book: RecipeBook):
        payoff = self._make_payoff(recipe_book)
        e1, e2 = payoff._parse_combination("")
        assert e1 is None
        assert e2 is None

    def test_parse_combination_long_narrative_returns_none(
        self, recipe_book: RecipeBook
    ):
        payoff = self._make_payoff(recipe_book)
        long_text = "x" * 250
        e1, e2 = payoff._parse_combination(long_text)
        assert e1 is None
        assert e2 is None

    def test_parse_combination_narrative_fallback(
        self, recipe_book: RecipeBook
    ):
        """If known element names appear in shorter narrative text, extract them."""
        payoff = self._make_payoff(recipe_book)
        text = "I want to try combining water and fire to see what happens."
        e1, e2 = payoff._parse_combination(text)
        # Should find the two elements via separator or fallback scan
        assert e1 is not None and e2 is not None

    def test_parse_combination_plus_separator(self, recipe_book: RecipeBook):
        payoff = self._make_payoff(recipe_book)
        e1, e2 = payoff._parse_combination("earth + fire")
        assert e1 == "earth"
        assert e2 == "fire"

    def test_parse_combination_with_separator(self, recipe_book: RecipeBook):
        payoff = self._make_payoff(recipe_book)
        e1, e2 = payoff._parse_combination("earth with fire")
        assert e1 == "earth"
        assert e2 == "fire"

    # -- metrics --------------------------------------------------------------

    def test_unique_discoveries_starts_at_zero(self, recipe_book: RecipeBook):
        """Only base elements exist at start, so unique_discoveries == 0."""
        payoff = self._make_payoff(recipe_book)
        assert payoff.unique_discoveries() == 0

    def test_knowledge_diversity_equal_inventories(
        self, recipe_book: RecipeBook
    ):
        """When all agents have identical inventories, diversity == 1.0."""
        payoff = self._make_payoff(recipe_book, player_names=["A", "B", "C"])
        # All start with the same 4 elements
        div = payoff.knowledge_diversity()
        assert div == pytest.approx(1.0)

    def test_knowledge_diversity_unequal_inventories(
        self, recipe_book: RecipeBook
    ):
        """When one agent has far more, diversity < 1.0."""
        payoff = self._make_payoff(recipe_book, player_names=["A", "B"])
        # Give A many extra elements
        for i in range(20):
            payoff.inventories["A"].elements.add(f"extra_{i}")
        div = payoff.knowledge_diversity()
        assert div < 1.0

    def test_discovery_rate_empty_history(self, recipe_book: RecipeBook):
        payoff = self._make_payoff(recipe_book)
        assert payoff.discovery_rate() == 0.0

    def test_discovery_rate_after_discoveries(self, recipe_book: RecipeBook):
        """After a successful discovery, rate should be > 0."""
        payoff = self._make_payoff(recipe_book)
        # Simulate a combination step
        scores = payoff.action_to_scores({"Alice": "water, fire", "Bob": "earth, fire"})
        rate = payoff.discovery_rate()
        # Both should discover something new
        assert rate > 0.0

    def test_cumulative_scores_start_at_zero(self, recipe_book: RecipeBook):
        payoff = self._make_payoff(recipe_book)
        for name, score in payoff.cumulative_scores.items():
            assert score == 0.0

    def test_action_to_scores_new_discovery(self, recipe_book: RecipeBook):
        """A valid new combination should yield a positive score."""
        payoff = self._make_payoff(recipe_book)
        scores = payoff.action_to_scores({"Alice": "water, fire", "Bob": "water, earth"})
        # Alice discovers steam, Bob discovers mud -- both new
        assert scores["Alice"] > 0
        assert scores["Bob"] > 0

    def test_action_to_scores_invalid_combination(
        self, recipe_book: RecipeBook
    ):
        """An invalid combination should yield -0.1."""
        payoff = self._make_payoff(recipe_book)
        scores = payoff.action_to_scores({"Alice": "water, air", "Bob": "water, air"})
        # water+air may or may not be a valid recipe -- let's use something certain
        # Use two elements that definitely don't combine:
        payoff2 = self._make_payoff(recipe_book)
        scores2 = payoff2.action_to_scores(
            {"Alice": "xyzzy, plugh", "Bob": "water, fire"}
        )
        # xyzzy and plugh are not in inventory -> -0.1
        assert scores2["Alice"] == pytest.approx(-0.1)

    def test_action_to_scores_redundant_combination(
        self, recipe_book: RecipeBook
    ):
        """Combining to get something already known yields 0.0."""
        payoff = self._make_payoff(recipe_book)
        # First time: discover steam
        payoff.action_to_scores({"Alice": "water, fire", "Bob": "water, fire"})
        # Second time: steam is already known -> redundant
        scores = payoff.action_to_scores(
            {"Alice": "water, fire", "Bob": "water, fire"}
        )
        assert scores["Alice"] == pytest.approx(0.0)
        assert scores["Bob"] == pytest.approx(0.0)

    def test_global_bonus_for_first_discovery(self, recipe_book: RecipeBook):
        """First global discovery earns 3.0 (1 base + 2 bonus)."""
        payoff = self._make_payoff(recipe_book)
        scores = payoff.action_to_scores({"Alice": "water, fire", "Bob": "earth, water"})
        # Both discover something brand new globally -> 3.0 each
        assert scores["Alice"] == pytest.approx(3.0)
        assert scores["Bob"] == pytest.approx(3.0)

    def test_second_agent_same_element_gets_no_global_bonus(
        self, recipe_book: RecipeBook
    ):
        """If Alice discovers steam first, Bob gets only 1.0 for the same."""
        payoff = self._make_payoff(recipe_book)
        # Alice discovers steam first
        payoff.action_to_scores({"Alice": "water, fire", "Bob": "earth, fire"})
        # Bob now discovers steam (new to Bob but not globally)
        scores = payoff.action_to_scores(
            {"Alice": "earth, fire", "Bob": "water, fire"}
        )
        # Alice: earth+fire -> lava was already discovered by Alice in step 1? No,
        # Alice did water+fire in step 1, Bob did earth+fire -> lava in step 1.
        # Step 2: Alice does earth+fire -> lava (already discovered by Bob globally) -> 1.0
        # Step 2: Bob does water+fire -> steam (already discovered by Alice globally) -> 1.0
        assert scores["Alice"] == pytest.approx(1.0)
        assert scores["Bob"] == pytest.approx(1.0)

    def test_should_terminate_respects_max_steps(self, recipe_book: RecipeBook):
        players = ["Alice"]
        cm = ConnectivityManager(players, mode="isolated")
        payoff = InnovationPayoff(
            player_names=players,
            recipe_book=recipe_book,
            connectivity_manager=cm,
            max_steps=2,
        )
        assert not payoff.should_terminate({})
        payoff.action_to_scores({"Alice": "water, fire"})
        assert not payoff.should_terminate({})
        payoff.action_to_scores({"Alice": "earth, fire"})
        assert payoff.should_terminate({})

    def test_innovation_score_at_start(self, recipe_book: RecipeBook):
        """At start: no discoveries, no rate, diversity=1 -> score = 0.2*1 = 0.2."""
        payoff = self._make_payoff(recipe_book)
        score = payoff.innovation_score()
        assert score == pytest.approx(0.2, abs=0.01)

    def test_inventory_summary(self, recipe_book: RecipeBook):
        payoff = self._make_payoff(recipe_book)
        summary = payoff.get_inventory_summary("Alice")
        assert "Alice" in summary
        assert "4 elements" in summary

    def test_get_inventory_summary_unknown_player(
        self, recipe_book: RecipeBook
    ):
        payoff = self._make_payoff(recipe_book)
        summary = payoff.get_inventory_summary("Unknown")
        assert "no inventory" in summary
