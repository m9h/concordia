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

"""Collective Innovation: LLM agents playing Little Alchemy 2.

This simulation implements the experimental paradigm described in "Collective
Innovation in Groups of Large Language Models" (arXiv:2407.05377). Agents start
with four basic elements (water, fire, earth, air) and attempt to discover new
elements by proposing pairwise combinations each step. The key experimental
variables are:

  - Prompt structure: open-ended vs. targeted, single-agent vs. multi-agent
    information context.
  - Network connectivity: fully connected (one group), isolated (solo agents),
    or dynamic (subgroups with stochastic inter-group visits).

Each simulation step consists of two scenes:
  1. Sharing scene -- agents in the same network group share recent discoveries
     via free-form conversation.
  2. Combination scene -- each agent chooses two elements from their inventory
     to combine. The payoff engine checks the recipe book and updates
     inventories accordingly.

The paper's main finding is that dynamic connectivity (small groups with
occasional inter-group visits) outperforms both fully connected and isolated
configurations in total unique discoveries.
"""

import collections
import dataclasses
import json
import math
import os
import random
import types
from typing import Any, Callable, Mapping, Sequence
from collections import Counter

from absl import logging
from examples.games.collective_innovation import social_data
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.entity import basic
from concordia.prefabs.game_master import dialogic_and_dramaturgic
from concordia.prefabs.game_master import game_theoretic_and_dramaturgic
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.components.agent import memory as memory_component
from concordia.components import game_master as gm_components
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib
from concordia.utils import helper_functions
from concordia.utils import structured_logging


# =============================================================================
# Recipe Book -- loads and queries alchemy_data.json
# =============================================================================


class RecipeBook:
  """Loads Little Alchemy 2 recipes and provides look-up methods.

  The backing data file is ``alchemy_data.json``, located alongside this
  module.  Format: ``{"entities": {"element_name": {"id": N, "recipes":
  [[parent1, parent2], ...]}, ...}}``.  Each recipe is a pair of parent
  element names that combine to create the keyed element.

  The recipe book builds an order-independent lookup so that
  ``try_combine("water", "fire")`` and ``try_combine("fire", "water")``
  both return the same result.
  """

  def __init__(self, data_path: str | None = None):
    """Load recipes from JSON.

    Args:
      data_path: Optional explicit path to the JSON file.  When *None*
        (the default), the file ``alchemy_data.json`` is expected in the
        same directory as this module.
    """
    if data_path is None:
      data_path = os.path.join(
          os.path.dirname(os.path.abspath(__file__)), "alchemy_data.json"
      )

    with open(data_path, "r") as f:
      raw_data = json.load(f)

    entities = raw_data["entities"]

    # _recipes: frozenset of Counter items -> result name
    # Using Counter for order-independent matching.
    self._recipes: dict[frozenset, str] = {}
    self._recipe_tuples: list[tuple[str, str, str]] = []

    all_elements: set[str] = set()

    for result_name, info in entities.items():
      result = result_name.strip().lower()
      all_elements.add(result)
      for recipe_pair in info.get("recipes", []):
        if len(recipe_pair) != 2:
          continue
        p1 = recipe_pair[0].strip().lower()
        p2 = recipe_pair[1].strip().lower()

        key = frozenset(Counter([p1, p2]).items())
        # Multiple recipes can produce the same result; first wins per key
        if key not in self._recipes:
          self._recipes[key] = result
        self._recipe_tuples.append((p1, p2, result))

        all_elements.update([p1, p2])

    self._all_elements = all_elements
    self._num_entities = len(all_elements)
    self._num_recipes = len(self._recipe_tuples)

  # -- public API ----------------------------------------------------------

  def try_combine(self, elem1: str, elem2: str) -> str | None:
    """Check whether combining *elem1* and *elem2* produces a result.

    The check is case-insensitive and order-independent.

    Args:
      elem1: First element name.
      elem2: Second element name.

    Returns:
      The name of the resulting element, or *None* if no recipe exists.
    """
    key = frozenset(Counter([elem1.lower(), elem2.lower()]).items())
    return self._recipes.get(key)

  def get_all_recipes(self) -> list[tuple[str, str, str]]:
    """Return every recipe as a list of (parent1, parent2, child) tuples."""
    return list(self._recipe_tuples)

  def get_possible_discoveries(self, inventory: set[str]) -> set[str]:
    """Given a current *inventory*, return the set of elements discoverable.

    An element is discoverable if there exists at least one recipe whose
    two parents are both present in the inventory.
    """
    lower_inv = {e.lower() for e in inventory}
    possible: set[str] = set()
    for p1, p2, result in self._recipe_tuples:
      if p1 in lower_inv and p2 in lower_inv:
        possible.add(result)
    return possible

  @property
  def num_entities(self) -> int:
    """Total number of distinct element names across all recipes."""
    return self._num_entities

  @property
  def num_recipes(self) -> int:
    """Total number of recipes."""
    return self._num_recipes


# =============================================================================
# Inventory state -- per-agent element tracking
# =============================================================================


@dataclasses.dataclass
class InventoryState:
  """Tracks a single agent's discovered elements and attempt history.

  Attributes:
    elements: The set of element names this agent currently possesses.
    discovery_history: Chronological list of successful discoveries, stored as
      (step, elem1, elem2, result) tuples.
    failed_attempts: Chronological list of failed combination attempts, stored
      as (step, elem1, elem2) tuples.
  """

  elements: set[str] = dataclasses.field(default_factory=set)
  discovery_history: list[tuple[int, str, str, str]] = dataclasses.field(
      default_factory=list
  )
  failed_attempts: list[tuple[int, str, str]] = dataclasses.field(
      default_factory=list
  )


# =============================================================================
# Connectivity Manager -- network structures from the paper
# =============================================================================


class ConnectivityManager:
  """Implements the three network structures from the paper.

  The paper (arXiv:2407.05377) evaluates three connectivity conditions:
    * **fully_connected** -- all agents share a single group.
    * **isolated** -- each agent is alone in its own group.
    * **dynamic** -- agents are arranged in subgroup pairs.  Each step, with
      probability ``visit_prob`` an agent may visit an adjacent subgroup for
      ``visit_duration`` steps, carrying knowledge between groups.
  """

  def __init__(
      self,
      agents: Sequence[str],
      mode: str = "fully_connected",
      group_size: int = 4,
      visit_prob: float = 0.2,
      visit_duration: int = 50,
      rng: random.Random | None = None,
  ):
    """Initialise the connectivity manager.

    Args:
      agents: Ordered list of agent names.
      mode: One of ``"fully_connected"``, ``"isolated"``, ``"dynamic"``.
      group_size: Size of each subgroup (used only for ``"dynamic"`` mode).
      visit_prob: Per-step probability that an agent begins a visit to an
        adjacent subgroup (dynamic mode only).
      visit_duration: Number of steps a visit lasts (dynamic mode only).
      rng: Random number generator for reproducibility.
    """
    self._agents = list(agents)
    self._mode = mode
    self._group_size = group_size
    self._visit_prob = visit_prob
    self._visit_duration = visit_duration
    self._rng = rng or random.Random()

    # Static base groups for dynamic mode.
    # Split agents into subgroups of ``group_size``.
    self._base_groups: list[list[str]] = []
    for i in range(0, len(self._agents), self._group_size):
      self._base_groups.append(self._agents[i : i + self._group_size])

    # Visit tracking: agent_name -> (destination_group_index, steps_remaining)
    self._active_visits: dict[str, tuple[int, int]] = {}

    # Map each agent to its base group index for quick lookup.
    self._agent_base_group: dict[str, int] = {}
    for gidx, group in enumerate(self._base_groups):
      for name in group:
        self._agent_base_group[name] = gidx

  # -- public API ----------------------------------------------------------

  def get_groups(self, step: int) -> list[list[str]]:
    """Return the current groupings for the given step.

    Args:
      step: The current simulation step (0-indexed).

    Returns:
      A list of groups, where each group is a list of agent names.
    """
    if self._mode == "fully_connected":
      return self.fully_connected()
    elif self._mode == "isolated":
      return self.isolated()
    elif self._mode == "dynamic":
      return self.dynamic(step)
    else:
      raise ValueError(f"Unknown connectivity mode: {self._mode}")

  def fully_connected(self) -> list[list[str]]:
    """All agents in one group."""
    return [list(self._agents)]

  def isolated(self) -> list[list[str]]:
    """Each agent in its own group."""
    return [[agent] for agent in self._agents]

  def dynamic(self, step: int) -> list[list[str]]:
    """Dynamic connectivity with stochastic inter-group visits.

    Each step:
      1. Decrement remaining visit durations; end expired visits.
      2. For each non-visiting agent, with probability ``visit_prob``,
         start a visit to an adjacent subgroup (only one visit at a time).
      3. Compute effective groups by overlaying visits onto base groups.

    Args:
      step: Current step number (used for state evolution).

    Returns:
      Current effective groupings.
    """
    # 1. Expire finished visits.
    expired = [
        agent
        for agent, (_, remaining) in self._active_visits.items()
        if remaining <= 1
    ]
    for agent in expired:
      del self._active_visits[agent]

    # Decrement remaining durations for continuing visits.
    for agent in list(self._active_visits):
      dest, remaining = self._active_visits[agent]
      self._active_visits[agent] = (dest, remaining - 1)

    # 2. Possibly start new visits.
    for agent in self._agents:
      if agent in self._active_visits:
        continue  # already visiting
      if self._rng.random() < self._visit_prob:
        base_gidx = self._agent_base_group[agent]
        num_groups = len(self._base_groups)
        if num_groups < 2:
          continue  # nowhere to visit
        # Pick an adjacent group (wrap around).
        adjacent_options = []
        if base_gidx > 0:
          adjacent_options.append(base_gidx - 1)
        if base_gidx < num_groups - 1:
          adjacent_options.append(base_gidx + 1)
        if not adjacent_options:
          continue
        dest_gidx = self._rng.choice(adjacent_options)
        self._active_visits[agent] = (dest_gidx, self._visit_duration)

    # 3. Build effective groups.
    effective: list[set[str]] = [set(g) for g in self._base_groups]
    for agent, (dest_gidx, _) in self._active_visits.items():
      base_gidx = self._agent_base_group[agent]
      # Agent leaves base group and joins destination group.
      effective[base_gidx].discard(agent)
      effective[dest_gidx].add(agent)

    # Convert to sorted lists, drop empty groups.
    return [sorted(g) for g in effective if g]


# =============================================================================
# Innovation Payoff -- the game engine
# =============================================================================


class InnovationPayoff:
  """Tracks innovation game state and computes scores.

  Scoring follows the paper:
    * Successful new discovery (new to this agent): +1.0
    * Successful discovery new to the *entire group*: +2.0 bonus
    * Redundant combination (result already known to agent): +0.0
    * Invalid combination (no recipe exists): -0.1
  """

  def __init__(
      self,
      player_names: Sequence[str],
      recipe_book: RecipeBook,
      connectivity_manager: ConnectivityManager,
      prompt_mode: str = "openended_multi",
      max_steps: int = 200,
  ):
    """Initialise the payoff engine.

    Args:
      player_names: List of participating agent names.
      recipe_book: The loaded recipe book instance.
      connectivity_manager: Manages network groupings.
      prompt_mode: One of the prompt condition labels from social_data.
      max_steps: Maximum number of combination steps before termination.
    """
    self._player_names = list(player_names)
    self.recipe_book = recipe_book
    self._connectivity = connectivity_manager
    self._prompt_mode = prompt_mode
    self._max_steps = max_steps
    self.current_step = 0

    # Per-agent inventories, each seeded with starting elements.
    self.inventories: dict[str, InventoryState] = {}
    for name in self._player_names:
      self.inventories[name] = InventoryState(
          elements=set(social_data.STARTING_ITEMS),
      )

    # Global tracking.
    self.global_discoveries: set[str] = set(social_data.STARTING_ITEMS)
    self.step_history: list[dict[str, Any]] = []
    self._latest_joint_action: dict[str, str] = {}
    self._cumulative_scores: dict[str, float] = {
        n: 0.0 for n in player_names
    }

  # -- core interface methods (called by the GM's PayoffMatrix component) ---

  @property
  def latest_joint_action(self) -> Mapping[str, str]:
    return self._latest_joint_action

  @property
  def cumulative_scores(self) -> Mapping[str, float]:
    return dict(self._cumulative_scores)

  def should_terminate(self, joint_action: Mapping[str, str]) -> bool:
    """Return True when the maximum number of steps has been reached."""
    return self.current_step >= self._max_steps

  def action_to_scores(
      self, joint_action: Mapping[str, str]
  ) -> Mapping[str, float]:
    """Parse each agent's combination attempt and update inventories.

    The *joint_action* maps ``agent_name -> "element1, element2"`` (a
    free-text string).  We parse the two element names, look them up in the
    recipe book, and assign scores following the paper's reward structure.

    Args:
      joint_action: Mapping of agent name to their chosen action string.

    Returns:
      Mapping of agent name to the score earned this step.
    """
    self._latest_joint_action = dict(joint_action)
    self.current_step += 1
    scores: dict[str, float] = {}
    step_record: dict[str, Any] = {
        "step": self.current_step,
        "actions": dict(joint_action),
        "results": {},
    }

    for player in self._player_names:
      action_str = joint_action.get(player, "")
      elem1, elem2 = self._parse_combination(action_str)

      if elem1 is None or elem2 is None:
        # Could not parse a valid pair.
        scores[player] = -0.1
        self.inventories[player].failed_attempts.append(
            (self.current_step, action_str, "")
        )
        step_record["results"][player] = {
            "status": "parse_error",
            "action": action_str,
        }
        continue

      # Normalise to lower case.
      elem1 = elem1.lower().strip()
      elem2 = elem2.lower().strip()

      # Check that both elements are in the agent's inventory.
      inv = self.inventories[player]
      inv_lower = {e.lower() for e in inv.elements}
      if elem1 not in inv_lower or elem2 not in inv_lower:
        scores[player] = -0.1
        inv.failed_attempts.append((self.current_step, elem1, elem2))
        step_record["results"][player] = {
            "status": "not_in_inventory",
            "elem1": elem1,
            "elem2": elem2,
        }
        continue

      result = self.recipe_book.try_combine(elem1, elem2)

      if result is None:
        # Invalid combination -- no recipe exists.
        scores[player] = -0.1
        inv.failed_attempts.append((self.current_step, elem1, elem2))
        step_record["results"][player] = {
            "status": "invalid",
            "elem1": elem1,
            "elem2": elem2,
        }
      elif result.lower() in {e.lower() for e in inv.elements}:
        # Redundant -- agent already knows this element.
        scores[player] = 0.0
        step_record["results"][player] = {
            "status": "redundant",
            "elem1": elem1,
            "elem2": elem2,
            "result": result,
        }
      else:
        # Genuine new discovery for this agent.
        score = 1.0

        # Bonus if the discovery is new to the entire group.
        if result.lower() not in {e.lower() for e in self.global_discoveries}:
          score += 2.0

        scores[player] = score
        inv.elements.add(result)
        inv.discovery_history.append(
            (self.current_step, elem1, elem2, result)
        )
        self.global_discoveries.add(result)
        step_record["results"][player] = {
            "status": "discovery",
            "elem1": elem1,
            "elem2": elem2,
            "result": result,
            "score": score,
        }

      self._cumulative_scores[player] += scores.get(player, 0.0)

    step_record["scores"] = dict(scores)
    self.step_history.append(step_record)
    return scores

  def scores_to_observation(
      self, scores: Mapping[str, float]
  ) -> Mapping[str, str]:
    """Generate natural language observations of combination results.

    For each agent, describe what happened: a new discovery, a redundant
    combination, or a failed attempt.  Also provide a summary of the agent's
    current inventory size and recent group discoveries.

    Args:
      scores: The scores computed by ``action_to_scores``.

    Returns:
      Mapping of agent name to a descriptive observation string.
    """
    if not self.step_history:
      return {player: "" for player in self._player_names}

    latest = self.step_history[-1]
    results_map = latest.get("results", {})
    observations: dict[str, str] = {}

    # Determine current groups for shared-discovery summaries.
    groups = self._connectivity.get_groups(self.current_step)
    agent_to_group: dict[str, int] = {}
    for gidx, group in enumerate(groups):
      for name in group:
        agent_to_group[name] = gidx

    for player in self._player_names:
      parts: list[str] = []
      result_info = results_map.get(player, {})
      status = result_info.get("status", "unknown")

      if status == "discovery":
        elem1 = result_info["elem1"]
        elem2 = result_info["elem2"]
        result = result_info["result"]
        sc = result_info.get("score", 1.0)
        parts.append(
            f"{player} combined {elem1} and {elem2} and discovered "
            f"{result}!"
        )
        if sc > 1.0:
          parts.append(
              f"This is a brand-new discovery that no one in the group "
              f"had found before. Bonus reward earned!"
          )
      elif status == "redundant":
        elem1 = result_info["elem1"]
        elem2 = result_info["elem2"]
        result = result_info["result"]
        parts.append(
            f"{player} combined {elem1} and {elem2}, which produces "
            f"{result}, but {player} already has {result}. No new "
            f"discovery this step."
        )
      elif status == "invalid":
        elem1 = result_info["elem1"]
        elem2 = result_info["elem2"]
        parts.append(
            f"{player} tried to combine {elem1} and {elem2}, but that "
            f"combination does not produce anything. A small penalty "
            f"was applied."
        )
      elif status == "not_in_inventory":
        elem1 = result_info.get("elem1", "?")
        elem2 = result_info.get("elem2", "?")
        parts.append(
            f"{player} tried to combine {elem1} and {elem2}, but at "
            f"least one of those elements is not in their inventory."
        )
      elif status == "parse_error":
        parts.append(
            f"{player}'s combination attempt could not be understood. "
            f"Please respond with exactly two element names separated "
            f"by a comma."
        )
      else:
        parts.append(f"{player} did not make a combination this step.")

      # Inventory summary.
      inv = self.inventories[player]
      parts.append(
          f"{player} now has {len(inv.elements)} elements in their "
          f"inventory."
      )

      # Group discovery summary.
      gidx = agent_to_group.get(player)
      if gidx is not None:
        group_members = groups[gidx]
        recent_group_discoveries = []
        for member in group_members:
          if member == player:
            continue
          member_result = results_map.get(member, {})
          if member_result.get("status") == "discovery":
            recent_group_discoveries.append(
                f"{member} discovered {member_result['result']}"
            )
        if recent_group_discoveries:
          parts.append(
              f"Meanwhile, group members also made discoveries: "
              f"{'; '.join(recent_group_discoveries)}."
          )

      observations[player] = " ".join(parts)

    return observations

  # -- helper methods ------------------------------------------------------

  def _parse_combination(
      self, action_str: str
  ) -> tuple[str | None, str | None]:
    """Parse a free-text action into two element names.

    Expected format: ``"element1, element2"`` (comma-separated).

    Args:
      action_str: The raw action string from the agent.

    Returns:
      A tuple ``(elem1, elem2)`` or ``(None, None)`` on parse failure.
    """
    if not action_str:
      return None, None

    # Remove common prefixes an LLM might add.
    cleaned = action_str.strip()
    for prefix in ["I choose ", "I combine ", "I want to combine ",
                    "Let me try ", "My choice: ", "Combining "]:
      if cleaned.lower().startswith(prefix.lower()):
        cleaned = cleaned[len(prefix):]
        break

    # Split on comma, " and ", or " + ".
    parts = None
    for sep in [",", " and ", " + "]:
      if sep in cleaned:
        parts = [p.strip().strip("'\"") for p in cleaned.split(sep, 1)]
        break

    if parts is None or len(parts) != 2:
      return None, None

    elem1, elem2 = parts[0].strip(), parts[1].strip()
    if not elem1 or not elem2:
      return None, None

    return elem1, elem2

  # -- metrics -------------------------------------------------------------

  def unique_discoveries(self) -> int:
    """Total unique elements discovered across all agents."""
    return len(self.global_discoveries) - len(social_data.STARTING_ITEMS)

  def discovery_rate(self, window: int = 10) -> float:
    """Average new discoveries per step over the last *window* steps."""
    if not self.step_history:
      return 0.0
    recent = self.step_history[-window:]
    total_new = 0
    for record in recent:
      for result_info in record.get("results", {}).values():
        if result_info.get("status") == "discovery":
          total_new += 1
    return total_new / len(recent)

  def knowledge_diversity(self) -> float:
    """Normalised Shannon entropy of per-agent discovery counts.

    Returns a value in [0, 1].  A value near 1 means discoveries are
    evenly spread across agents; near 0 means one agent holds most
    knowledge.
    """
    counts = [len(inv.elements) for inv in self.inventories.values()]
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
      return 1.0

    probs = [c / total for c in counts if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(counts))
    if max_entropy == 0:
      return 1.0
    return entropy / max_entropy

  def innovation_score(self) -> float:
    """Composite metric combining discovery count, rate, and diversity.

    ``score = 0.5 * normalised_discoveries + 0.3 * rate + 0.2 * diversity``

    The discovery count is normalised against the total number of recipes
    in the book (as a rough upper bound on discoverable elements).
    """
    norm_disc = (
        self.unique_discoveries() / max(self.recipe_book.num_recipes, 1)
    )
    rate = min(self.discovery_rate() / max(len(self._player_names), 1), 1.0)
    div = self.knowledge_diversity()
    return 0.5 * norm_disc + 0.3 * rate + 0.2 * div

  def get_inventory_summary(self, player: str) -> str:
    """Return a human-readable summary of a player's inventory."""
    inv = self.inventories.get(player)
    if inv is None:
      return f"{player} has no inventory."
    sorted_elems = sorted(inv.elements)
    return (
        f"{player} has {len(sorted_elems)} elements: "
        f"{', '.join(sorted_elems)}."
    )

  def get_shared_discoveries_for(self, player: str) -> str:
    """Return a summary of recent group discoveries visible to *player*.

    Used to populate the ``{shared_discoveries}`` template variable in
    multi-agent prompt conditions.
    """
    groups = self._connectivity.get_groups(self.current_step)
    # Find player's group.
    group_members: list[str] = []
    for group in groups:
      if player in group:
        group_members = group
        break

    recent: list[str] = []
    for member in group_members:
      if member == player:
        continue
      inv = self.inventories[member]
      # Last 3 discoveries from this member.
      for _, e1, e2, result in inv.discovery_history[-3:]:
        recent.append(f"{member} found {result} (from {e1} + {e2})")

    if not recent:
      return "No recent group discoveries."
    return "; ".join(recent)


# =============================================================================
# Custom Game Master prefabs with payoff-based termination
# =============================================================================


_CURRENT_PAYOFF: InnovationPayoff | None = None


class PayoffBasedTerminator(entity_component.ComponentWithLogging):
  """Terminates the simulation when the payoff engine's step limit is hit."""

  def __init__(
      self,
      payoff: InnovationPayoff,
      scene_tracker: entity_component.ContextComponent,
  ):
    super().__init__()
    self._payoff = payoff
    self._scene_tracker = scene_tracker

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type == entity_lib.OutputType.TERMINATE:
      if self._payoff.should_terminate({}):
        return entity_lib.BINARY_OPTIONS["affirmative"]
      return entity_lib.BINARY_OPTIONS["negative"]
    if action_spec.output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
      if self._payoff.should_terminate({}):
        return "None"
      return self._scene_tracker.pre_act(action_spec)
    return ""

  def get_participants(self) -> Sequence[str]:
    if hasattr(self._scene_tracker, "get_participants"):
      return self._scene_tracker.get_participants()
    return []

  def get_current_scene_type(self) -> scene_lib.SceneTypeSpec:
    return self._scene_tracker.get_current_scene_type()

  def get_num_scenes(self) -> int:
    return self._scene_tracker.get_num_scenes()

  def get_current_scene_index(self) -> int:
    return self._scene_tracker.get_current_scene_index()

  def get_last_log(self) -> Mapping[str, Any]:
    if hasattr(self._scene_tracker, "get_last_log"):
      return self._scene_tracker.get_last_log()
    return {}

  def get_pre_act_value(self) -> str:
    return ""

  def get_pre_act_label(self) -> str:
    return ""

  def post_act(self, event_statement: str) -> str:
    return ""

  def pre_observe(self, observation: str | None = None) -> str:
    return ""

  def post_observe(self, observation: str | None = None) -> str:
    return ""

  def update(self) -> None:
    pass

  def get_state(self) -> entity_component.ComponentState:
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    pass

  def terminate(self) -> None:
    self._terminate_now = True


class CustomConversationGM(dialogic_and_dramaturgic.GameMaster):
  """Conversation GM with payoff-based termination."""

  def build(self, model, memory_bank):
    gm = super().build(model, memory_bank)
    if _CURRENT_PAYOFF:
      scene_tracker_key = (
          gm_components.next_game_master
          .DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
      )
      original_scene_tracker = gm._context_components[scene_tracker_key]
      terminator = PayoffBasedTerminator(
          _CURRENT_PAYOFF, original_scene_tracker
      )
      gm._context_components[
          gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
      ] = terminator
      gm._context_components[scene_tracker_key] = terminator
    return gm


class CustomDecisionGM(game_theoretic_and_dramaturgic.GameMaster):
  """Decision GM with payoff-based termination."""

  def build(self, model, memory_bank):
    gm = super().build(model, memory_bank)
    if _CURRENT_PAYOFF:
      scene_tracker_key = (
          gm_components.next_game_master
          .DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
      )
      original_scene_tracker = gm._context_components[scene_tracker_key]
      terminator = PayoffBasedTerminator(
          _CURRENT_PAYOFF, original_scene_tracker
      )
      gm._context_components[
          gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
      ] = terminator
      gm._context_components[scene_tracker_key] = terminator
    return gm


# =============================================================================
# Scene configuration
# =============================================================================


def configure_scenes(
    people: Sequence[str],
    payoff: InnovationPayoff,
    connectivity: ConnectivityManager,
    num_steps: int,
    skip_conversation: bool = False,
) -> Sequence[scene_lib.SceneSpec]:
  """Build the sequence of scenes for the entire simulation.

  Each simulation step produces:
    1. A **sharing scene** (conversation) where agents in the same network
       group share recent discoveries via free-form speech.
    2. A **combination scene** (decision) where each agent chooses two
       elements to combine.

  Args:
    people: List of all agent names.
    payoff: The innovation payoff engine (used for inventory summaries).
    connectivity: Manages dynamic group membership.
    num_steps: Total number of combination steps.
    skip_conversation: If True, omit sharing scenes (faster runs).

  Returns:
    A sequence of ``SceneSpec`` instances to pass to the GM prefabs.
  """
  scenes: list[scene_lib.SceneSpec] = []

  for step_idx in range(num_steps):
    step_num = step_idx + 1
    groups = connectivity.get_groups(step_idx)

    # ------------------------------------------------------------------
    # 1. Sharing scene (conversation) -- one per group
    # ------------------------------------------------------------------
    if not skip_conversation:
      for gidx, group in enumerate(groups):
        if len(group) <= 1:
          # No point having a sharing scene with one agent.
          continue

        sharing_scene_type = scene_lib.SceneTypeSpec(
            name=f"step_{step_num}_sharing_g{gidx}",
            game_master_name="conversation rules",
            action_spec=entity_lib.free_action_spec(
                call_to_action=social_data.SHARE_CALL_TO_SPEECH,
                tag="sharing",
            ),
        )

        premise: dict[str, list[str | Callable]] = {}
        for name in group:
          inv = payoff.inventories[name]
          elem_list = ", ".join(sorted(inv.elements))
          recent_disc = (
              payoff.get_shared_discoveries_for(name)
          )
          premise[name] = [
              social_data.SHARE_DISCOVERY_PREMISE.format(
                  round_num=step_num
              ),
              f"{name}'s current elements: {elem_list}.",
              f"Recent group discoveries: {recent_disc}.",
          ]

        scenes.append(
            scene_lib.SceneSpec(
                scene_type=sharing_scene_type,
                participants=list(group),
                num_rounds=len(group),
                premise=premise,
            )
        )

    # ------------------------------------------------------------------
    # 2. Combination scene (decision) -- all agents act simultaneously
    # ------------------------------------------------------------------
    combination_scene_type = scene_lib.SceneTypeSpec(
        name=f"step_{step_num}_combine",
        game_master_name="decision rules",
        action_spec=entity_lib.free_action_spec(
            call_to_action=(
                "Which two elements does {name} combine? "
                "Respond with exactly two element names from your "
                "inventory, separated by a comma (e.g., 'water, fire')."
            ),
            tag="combination",
        ),
    )

    combination_premise: dict[str, list[str | Callable]] = {}
    for name in people:
      inv = payoff.inventories[name]
      elem_list = ", ".join(sorted(inv.elements))
      num_elems = len(inv.elements)

      premise_parts: list[str | Callable] = [
          (
              f"Step {step_num} of the alchemy experiment. {name} has "
              f"{num_elems} elements: {elem_list}."
          ),
      ]

      # Add shared discoveries context for multi-agent prompts.
      shared_disc = payoff.get_shared_discoveries_for(name)
      if shared_disc and "No recent" not in shared_disc:
        premise_parts.append(
            f"Recent group discoveries: {shared_disc}."
        )

      # Add failed-attempt context (last 3 failures).
      recent_failures = inv.failed_attempts[-3:]
      if recent_failures:
        fail_strs = [
            f"{e1} + {e2}" for _, e1, e2 in recent_failures
        ]
        premise_parts.append(
            f"Recent failed combinations: {', '.join(fail_strs)}."
        )

      combination_premise[name] = premise_parts

    scenes.append(
        scene_lib.SceneSpec(
            scene_type=combination_scene_type,
            participants=list(people),
            num_rounds=len(people),
            premise=combination_premise,
        )
    )

  return scenes


# =============================================================================
# Main simulation runner
# =============================================================================


def run_simulation(
    model: language_model.LanguageModel,
    embedder: Callable[[str], Any],
    num_steps: int = 200,
    seed: int | None = None,
    num_agents: int = 8,
    connectivity: str = "fully_connected",
    prompt_mode: str = "openended_multi",
    group_size: int = 4,
    visit_prob: float = 0.2,
    visit_duration: int = 50,
    skip_backstory: bool = False,
    skip_conversation: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
  """Run the Collective Innovation simulation.

  This is the main entry point.  It:
    1. Selects agents from the profile catalogue.
    2. Initialises the RecipeBook and per-agent inventories.
    3. Creates a ConnectivityManager for the chosen network structure.
    4. Configures scenes for every step.
    5. Creates the InnovationPayoff engine.
    6. Builds the Concordia prefab/instance configuration.
    7. Runs ``simulation_lib.Simulation(config, model, embedder).play()``.
    8. Compiles and returns a results dictionary.

  Args:
    model: The language model for agent cognition.
    embedder: Sentence embedder for associative memory.
    num_steps: Number of combination steps (default 200 per the paper).
    seed: Random seed for reproducibility.
    num_agents: Number of agents to use (sampled from AGENT_PROFILES).
    connectivity: Network mode -- ``"fully_connected"``, ``"isolated"``,
      or ``"dynamic"``.
    prompt_mode: Prompt condition label from ``social_data.PROMPT_CONDITIONS``.
    group_size: Subgroup size for dynamic connectivity.
    visit_prob: Visit probability for dynamic connectivity.
    visit_duration: Visit duration (steps) for dynamic connectivity.
    skip_backstory: If True, skip the formative-memories initialiser.
    skip_conversation: If True, skip sharing (conversation) scenes.
    verbose: If True, set logging to INFO level.

  Returns:
    A dictionary containing simulation results: scores, discoveries,
    metrics, and the structured log.
  """
  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)

  # ---- select agents -----------------------------------------------------
  all_names = list(social_data.AGENT_PROFILES.keys())
  if num_agents > len(all_names):
    num_agents = len(all_names)
  agents_to_use = rng.sample(all_names, num_agents)
  people = list(agents_to_use)

  # ---- initialise recipe book --------------------------------------------
  recipe_book = RecipeBook()

  # ---- initialise connectivity manager -----------------------------------
  conn_config = social_data.CONNECTIVITY_CONFIGS.get(connectivity, {})
  effective_group_size = conn_config.get("group_size", group_size) or group_size
  effective_visit_prob = conn_config.get("visit_prob", visit_prob)
  effective_visit_duration = conn_config.get("visit_duration", visit_duration)

  conn_manager = ConnectivityManager(
      agents=people,
      mode=connectivity,
      group_size=effective_group_size,
      visit_prob=effective_visit_prob,
      visit_duration=effective_visit_duration,
      rng=rng,
  )

  # ---- initialise payoff engine ------------------------------------------
  payoff = InnovationPayoff(
      player_names=people,
      recipe_book=recipe_book,
      connectivity_manager=conn_manager,
      prompt_mode=prompt_mode,
      max_steps=num_steps,
  )

  global _CURRENT_PAYOFF
  _CURRENT_PAYOFF = payoff

  # ---- configure scenes --------------------------------------------------
  scenes = configure_scenes(
      people=people,
      payoff=payoff,
      connectivity=conn_manager,
      num_steps=num_steps,
      skip_conversation=skip_conversation,
  )

  # ---- build prefabs and instances ---------------------------------------
  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(game_master_prefabs),
  }

  # Register custom GM prefabs.
  prefabs["conversation_rules__GameMaster"] = CustomConversationGM()
  prefabs["decision_rules__GameMaster"] = CustomDecisionGM()

  instances: list[prefab_lib.InstanceConfig] = []
  player_specific_memories: dict[str, list[str]] = {}

  for name in people:
    profile = social_data.AGENT_PROFILES[name]
    archetype = profile["archetype"]

    goal = (
        f"Discover as many new elements as possible in the alchemy "
        f"experiment. As a {archetype.value} type, {name} approaches "
        f"discovery in their own way. {name} should make strategic use "
        f"of information shared by group members and try combinations "
        f"that are likely to yield new results."
    )

    instances.append(
        prefab_lib.InstanceConfig(
            prefab="basic__Entity",
            role=prefab_lib.Role.ENTITY,
            params={
                "name": name,
                "goal": goal,
            },
        )
    )

    # Build agent-specific memories.
    elem_list = ", ".join(sorted(social_data.STARTING_ITEMS))
    memories = [
        (
            f"{name} is participating in a Little Alchemy 2 experiment "
            f"where the goal is to discover new elements by combining "
            f"existing ones."
        ),
        f"{name}'s starting elements are: {elem_list}.",
        profile["backstory"],
        f"{name}'s personality: {profile['personality']}",
        (
            f"{name}'s archetype is {archetype.value}: this influences "
            f"how they approach element combination."
        ),
        f"[goal] {goal}",
    ]
    player_specific_memories[name] = memories

  # Shared world memories.
  shared_memories = [
      (
          "This is a Little Alchemy 2 experiment. Participants start with "
          "four basic elements: water, fire, earth, and air. By combining "
          "two elements, a new element may be discovered. Not all "
          "combinations produce results."
      ),
      (
          "The experiment runs for many rounds. Each round, participants "
          "choose two elements from their inventory to combine. If the "
          "combination is valid, the resulting element is added to their "
          "inventory."
      ),
      (
          "Participants may share their discoveries with group members "
          "during sharing sessions. Information from others can help "
          "guide future combination choices."
      ),
      (
          "The goal is collective: the group aims to discover as many "
          "unique elements as possible. Both individual exploration and "
          "strategic information sharing contribute to group success."
      ),
  ]

  # Memory initialiser (slow -- skip for fast testing).
  if not skip_backstory:
    instances.append(
        prefab_lib.InstanceConfig(
            prefab="formative_memories_initializer__GameMaster",
            role=prefab_lib.Role.INITIALIZER,
            params={
                "name": "initial setup rules",
                "next_game_master_name": "conversation rules",
                "shared_memories": shared_memories,
                "player_specific_memories": player_specific_memories,
            },
        )
    )

  # Conversation GM (for sharing scenes).
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="conversation_rules__GameMaster",
          role=(
              prefab_lib.Role.INITIALIZER
              if skip_backstory
              else prefab_lib.Role.GAME_MASTER
          ),
          params={
              "name": "conversation rules",
              "scenes": scenes,
          },
      )
  )

  # Decision GM (for combination scenes with payoff matrix).
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="decision_rules__GameMaster",
          role=prefab_lib.Role.GAME_MASTER,
          params={
              "name": "decision rules",
              "scenes": scenes,
              "action_to_scores": payoff.action_to_scores,
              "scores_to_observation": payoff.scores_to_observation,
          },
      )
  )

  # ---- assemble config ---------------------------------------------------
  config = prefab_lib.Config(
      default_premise=(
          "A group of participants is taking part in a Little Alchemy 2 "
          "experiment. They each start with four basic elements -- water, "
          "fire, earth, and air -- and must discover as many new elements "
          "as possible by combining pairs. The experiment tests how "
          "different communication structures affect collective innovation. "
          "Participants can share their discoveries during group sessions "
          "and must decide strategically which combinations to try."
      ),
      default_max_steps=len(scenes) + 5,
      prefabs=prefabs,
      instances=instances,
  )

  # ---- run ---------------------------------------------------------------
  if verbose:
    logging.set_verbosity(logging.INFO)
  else:
    logging.set_verbosity(logging.ERROR)

  sim = simulation_lib.Simulation(
      config=config,
      model=model,
      embedder=embedder,
  )

  force_steps = len(scenes) + 1
  structured_log = sim.play(force_steps=force_steps)

  # ---- compile results ---------------------------------------------------
  per_agent_results: dict[str, dict[str, Any]] = {}
  for name in people:
    inv = payoff.inventories[name]
    per_agent_results[name] = {
        "num_elements": len(inv.elements),
        "elements": sorted(inv.elements),
        "num_discoveries": len(inv.discovery_history),
        "num_failed_attempts": len(inv.failed_attempts),
        "cumulative_score": payoff.cumulative_scores[name],
        "archetype": social_data.AGENT_PROFILES[name]["archetype"].value,
    }

  return {
      "seed": seed,
      "num_steps": num_steps,
      "num_agents": num_agents,
      "connectivity": connectivity,
      "prompt_mode": prompt_mode,
      "agent_names": people,
      "per_agent_results": per_agent_results,
      "cumulative_scores": dict(payoff.cumulative_scores),
      "unique_discoveries": payoff.unique_discoveries(),
      "global_discoveries": sorted(payoff.global_discoveries),
      "discovery_rate": payoff.discovery_rate(),
      "knowledge_diversity": payoff.knowledge_diversity(),
      "innovation_score": payoff.innovation_score(),
      "step_history": payoff.step_history,
      "structured_log": structured_log,
  }
