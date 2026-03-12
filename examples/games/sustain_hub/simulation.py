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

"""SustainHub: Open-source community sustainability simulation.

This simulation models an open-source software project where LLM agents with
distinct roles (Contributors, Innovators, Knowledge Curators, Maintainers)
must collectively manage a task queue across multiple sprints.

The core social dilemmas are:
  - Preferred vs. urgent: Take the task you're best at (+3 reward) or the
    one the project most needs (+1 reward, but prevents project decay)?
  - Mentoring vs. productivity: Spend time helping a newcomer (investment
    in future capacity) or maximize your own output?
  - Thorough vs. fast review: Maintainers can rubber-stamp PRs to clear the
    queue, or review carefully at the cost of throughput.
  - Individual recognition vs. collective health: Flashy features get
    attention; bug fixes and docs keep the lights on.

Inspired by the SustainHub framework (Rohira, 2025) which uses SARSA-based
reinforcement learning. Here we replace Q-tables with LLM cognition —
agents learn from their episodic memory of past sprint outcomes and adapt
their strategies through natural language reasoning.

Reference: https://vidhirohira.github.io/blog/2025/08/25/final-evaluation.html
"""

import collections
import dataclasses
import random
import types
from typing import Any, Callable, Mapping, Sequence

from absl import logging
from examples.games.sustain_hub import social_data
from examples.games.sustain_hub import tools as sustain_tools
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as game_master_prefabs
from concordia.prefabs.entity import basic
from concordia.prefabs.entity import rational
from concordia.prefabs.game_master import dialogic_and_dramaturgic
from concordia.prefabs.game_master import game_theoretic_and_dramaturgic
from concordia.prefabs.simulation import generic as simulation_lib
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components import game_master as gm_components
from concordia.document import interactive_document
from concordia.document import interactive_document_tools
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib
from concordia.utils import helper_functions
from concordia.utils import structured_logging


# =============================================================================
# Tool-using Agent Component
# =============================================================================


class QuestionOfRecentMemoriesWithTools(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A question that can use tools to gather information."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      pre_act_label: str,
      question: str,
      answer_prefix: str,
      tools: Sequence[sustain_tools.tool_lib.Tool] = (),
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      num_memories_to_retrieve: int = 25,
  ):
    super().__init__(pre_act_label)
    self._model = model
    self._tools = tools
    self._memory_component_key = memory_component_key
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._question = question
    self._answer_prefix = answer_prefix

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )

    prompt = interactive_document_tools.InteractiveDocumentWithTools(
        self._model, tools=self._tools
    )

    if self._num_memories_to_retrieve > 0:
      mems = '\n'.join([
          mem
          for mem in memory.retrieve_recent(
              limit=self._num_memories_to_retrieve
          )
      ])
      prompt.statement(f'Recent observations of {agent_name}:\n{mems}')

    question = self._question.format(agent_name=agent_name)
    result = prompt.open_question(
        question,
        answer_prefix=self._answer_prefix.format(agent_name=agent_name),
        max_tokens=1000,
    )
    result = self._answer_prefix.format(agent_name=agent_name) + result

    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Summary': question,
        'State': result,
        'Chain of thought': prompt.view().text().splitlines(),
    })

    return result


class ActiveInferenceSituationPerception(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A perception component based on Active Inference principles."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      num_memories_to_retrieve: int = 25,
  ):
    super().__init__('\nQuestion: How does the agent perceive their situation through the lens of Active Inference?\nAnswer')
    self._model = model
    self._memory_component_key = memory_component_key
    self._num_memories_to_retrieve = num_memories_to_retrieve

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )

    prompt = interactive_document.InteractiveDocument(self._model)

    if self._num_memories_to_retrieve > 0:
      mems = '\n'.join([
          mem
          for mem in memory.retrieve_recent(
              limit=self._num_memories_to_retrieve
          )
      ])
      prompt.statement(f'Recent observations of {agent_name}:\n{mems}')

    prompt.statement(f"Consider {agent_name}'s situation using the principles of Active Inference (minimizing Expected Free Energy):")
    
    # Pragmatic Value (Goal alignment)
    pragmatic = prompt.open_question(
        f"What are {agent_name}'s current 'priors' (goals and expectations) for the project's health and their own career? What actions would have the highest Pragmatic Value (directly achieving these goals)?",
        max_tokens=500,
    )
    
    # Epistemic Value (Uncertainty reduction)
    epistemic = prompt.open_question(
        f"Where does {agent_name} have the most uncertainty? (e.g., unfamiliar code, unknown teammate reliability, unclear project priorities). What actions would have the highest Epistemic Value (reducing this uncertainty through exploration or 'information foraging')?",
        max_tokens=500,
    )

    # Uncertainty Quantification (System 2 grounding)
    uncertainty_score = prompt.open_question(
        f"On a scale of 0-10 (where 0 is absolute certainty and 10 is maximum surprise/uncertainty), how 'surprising' or 'unpredictable' does the current community state feel to {agent_name}? Provide only the number.",
        max_tokens=10,
    )
    
    # Synthesis
    final_perception = prompt.open_question(
        f"Synthesize these assessments. Given an uncertainty score of {uncertainty_score}/10, how should {agent_name} balance goal-seeking (Pragmatic) vs. information-seeking (Epistemic) in the next sprint? State their overall strategy.",
        answer_prefix=f"{agent_name} is currently ",
        max_tokens=500,
    )

    result = f"{agent_name} is currently {final_perception} (Uncertainty Score: {uncertainty_score}/10)"

    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Pragmatic Assessment': pragmatic,
        'Epistemic Assessment': epistemic,
        'Uncertainty Score': uncertainty_score,
        'Strategy': result,
    })

    return result


@dataclasses.dataclass
class SustainHubEntity(basic.Entity):
  """A prefab for SustainHub agents that can use tools and Active Inference."""

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    agent = super().build(model, memory_bank)
    
    use_active_inference = self.params.get('use_active_inference', True)

    if use_active_inference:
      # Replace SituationPerception with Active Inference version
      situation_perception = ActiveInferenceSituationPerception(
          model=model,
          num_memories_to_retrieve=self.params.get(
              'situation_perception_history_length', 25),
      )
      agent._context_components['SituationPerception'] = situation_perception
      situation_perception.set_entity(agent)
    else:
      # Use the original tool-using version (or default)
      situation_perception = QuestionOfRecentMemoriesWithTools(
          model=model,
          pre_act_label=f'\nQuestion: What situation is {agent.name} in right now?\nAnswer',
          question='What kind of situation is {agent_name} in right now? You may use tools to check project health or team status.',
          answer_prefix='{agent_name} is currently ',
          num_memories_to_retrieve=self.params.get(
              'situation_perception_history_length', 25),
      )
      agent._context_components['SituationPerception'] = situation_perception
      situation_perception.set_entity(agent)
    
    # Tool use component
    tools = self.params.get('tools', [])
    if tools:
      # We still want agents to be able to use tools, so we add a specific component for it
      tool_component = QuestionOfRecentMemoriesWithTools(
          model=model,
          tools=tools,
          pre_act_label=f'\nQuestion: Which tools should {agent.name} use?\nAnswer',
          question='Which tools, if any, should {agent_name} use to gather more data or assist with tasks? You may check project stats or mentorship needs.',
          answer_prefix='{agent_name} decides to ',
      )
      agent._context_components['ToolSelection'] = tool_component
      tool_component.set_entity(agent)

    return agent


# =============================================================================
# Payoff engine
# =============================================================================


class SustainHubPayoff:
  """Computes payoffs for task selection decisions.

  Implements SustainHub's reward structure:
    - Preferred task type, success: +3
    - Non-preferred task type, success: +1
    - Any task, failure: -1
    - Skip: 0

  Also tracks the Harmony Index:
    HI = alpha * avg_success + (1 - alpha) * fairness_score
  """

  def __init__(
      self,
      player_names: Sequence[str],
      player_roles: Mapping[str, social_data.Role],
      task_options: Sequence[str],
      task_type_map: Mapping[str, str],
      relational_matrix: Mapping[str, Mapping[str, float]],
      num_sprints: int,
      alpha: float = 0.6,
      player_tools: Mapping[str, Sequence[sustain_tools.tool_lib.Tool]] = (
          types.MappingProxyType({})
      ),
  ):
    self._player_names = list(player_names)
    self._player_roles = player_roles
    self._task_options = task_options
    self._task_type_map = task_type_map  # maps task label -> task type
    self._relational_matrix = relational_matrix
    self._num_sprints = num_sprints
    self._alpha = alpha
    self._player_tools = player_tools
    self._latest_joint_action: dict[str, str] = {}
    self._cumulative_scores: dict[str, float] = {n: 0.0 for n in player_names}
    self._task_counts: dict[str, int] = {n: 0 for n in player_names}
    self._sprint_history: list[dict[str, Any]] = []
    self._prev_usage_counts = {p: 0 for p in player_names}
    self.current_policy = "None"
    self.policy_config = {
        "Tool Subsidy": {"success_prob_bonus": 0.2},
        "Maintenance Premium": {
            "reward_bonus": 1.0,
            "task_types": ["bug_fix", "documentation"],
        },
    }

  @property
  def latest_joint_action(self) -> Mapping[str, str]:
    return self._latest_joint_action

  @property
  def cumulative_scores(self) -> Mapping[str, float]:
    return dict(self._cumulative_scores)

  @property
  def sprint_history(self) -> list[dict[str, Any]]:
    return list(self._sprint_history)

  def should_terminate(self, joint_action: Mapping[str, str]) -> bool:
    return len(self._sprint_history) >= self._num_sprints

  @property
  def resilience_quotient(self) -> float:
    """Calculate the Resilience Quotient (RQ).
    
    RQ = HI_after_stress / HI_before_stress
    """
    if len(self._sprint_history) < 2:
      return 1.0
    
    # Find if a dropout happened
    dropout_sprint = -1
    for i, s in enumerate(self._sprint_history):
      if any(not s["joint_action"].get(p) for p in self._player_names):
        dropout_sprint = i
        break
    
    if dropout_sprint == -1:
      return 1.0 
      
    hi_before = sum(s["harmony_index"] for s in self._sprint_history[:dropout_sprint]) / dropout_sprint if dropout_sprint > 0 else 1.0
    hi_after = sum(s["harmony_index"] for s in self._sprint_history[dropout_sprint:]) / (len(self._sprint_history) - dropout_sprint)
    
    return min(1.0, hi_after / hi_before)

  def harmony_index(self) -> float:
    """Compute the Harmony Index: HI = alpha * avg_success + (1-alpha) * fairness."""
    if not self._task_counts or all(v == 0 for v in self._task_counts.values()):
      return 0.0

    # Average success rate (normalized)
    total_tasks = sum(self._task_counts.values())
    total_score = sum(self._cumulative_scores.values())
    max_possible = total_tasks * social_data.REWARD_PREFERRED_SUCCESS
    avg_success = total_score / max_possible if max_possible > 0 else 0.0

    # Fairness: 1 - Gini coefficient of task counts
    counts = sorted(self._task_counts.values())
    n = len(counts)
    if n == 0 or sum(counts) == 0:
      fairness = 1.0
    else:
      numerator = sum(
          abs(counts[i] - counts[j])
          for i in range(n) for j in range(n)
      )
      denominator = 2 * n * sum(counts)
      gini = numerator / denominator if denominator > 0 else 0.0
      fairness = 1.0 - gini

    return self._alpha * avg_success + (1.0 - self._alpha) * fairness

  def action_to_scores(
      self, joint_action: Mapping[str, str]
  ) -> Mapping[str, float]:
    """Map joint task selections to individual scores."""
    self._latest_joint_action = dict(joint_action)

    # Check if this is a policy vote
    policy_options = [
        "Continue as-is",
        "Implement Tool Subsidy",
        "Implement Maintenance Premium",
    ]
    if any(choice in policy_options for choice in joint_action.values()):
      votes = [c for c in joint_action.values() if c in policy_options]
      if votes:
        majority_vote = collections.Counter(votes).most_common(1)[0][0]
        if majority_vote == "Implement Tool Subsidy":
          self.current_policy = "Tool Subsidy"
        elif majority_vote == "Implement Maintenance Premium":
          self.current_policy = "Maintenance Premium"
      return {player: 0.0 for player in self._player_names}

    scores: dict[str, float] = {}

    # Track how many people chose each task (overloading penalty)
    task_choosers: dict[str, list[str]] = collections.defaultdict(list)
    for player, task in joint_action.items():
      task_choosers[task].append(player)

    for player in self._player_names:
      chosen_task = joint_action.get(player, "")

      # Skip case
      if not chosen_task or chosen_task.lower().startswith("skip"):
        scores[player] = social_data.REWARD_SKIP
        continue

      # Determine task type
      task_type = self._task_type_map.get(chosen_task, "unknown")
      preferred_type = social_data.ROLE_PREFERRED_TASKS.get(
          self._player_roles.get(player, social_data.Role.CONTRIBUTOR),
          "bug_fix",
      )
      is_preferred = (task_type == preferred_type)

      # Success probability based on expertise level and task alignment
      # Calibrated from empirical OSS data (Gemini research brief):
      #   Apprentice: 25%, Regular/Intermediate: 75%, Expert/Senior: 95%
      expertise = social_data.AGENT_PROFILES.get(player, {}).get(
          'expertise', social_data.ExpertiseLevel.INTERMEDIATE)
      expertise_probs = {
          social_data.ExpertiseLevel.APPRENTICE: 0.25,
          social_data.ExpertiseLevel.INTERMEDIATE: 0.55,
          social_data.ExpertiseLevel.SENIOR: 0.75,
          social_data.ExpertiseLevel.EXPERT: 0.85,
      }
      base_prob = expertise_probs.get(expertise, 0.55)
      # Preferred task bonus
      if is_preferred:
        base_prob += 0.10

      # Collaboration bonus: if a friend chose the same task, teamwork bonus
      collab_bonus = 0.0
      for other in task_choosers[chosen_task]:
        if other != player:
          rel = self._relational_matrix.get(player, {}).get(other, 0.0)
          collab_bonus += 0.05 * rel

      # Check if an agent used AutoCodeRover tool (usage_count increased)
      tool_bonus = 0.0
      tools = self._player_tools.get(player, [])
      for tool in tools:
        if isinstance(tool, sustain_tools.AutoCodeRover):
          if tool.usage_count > self._prev_usage_counts[player]:
            tool_bonus = 0.15
            self._prev_usage_counts[player] = tool.usage_count
          break

      # Overload penalty: too many people on one task is wasteful
      num_on_task = len(task_choosers[chosen_task])
      overload_penalty = max(0.0, (num_on_task - 2) * 0.1)

      success_prob = base_prob + collab_bonus + tool_bonus - overload_penalty

      # Apply Tool Subsidy Policy
      if self.current_policy == "Tool Subsidy" and tool_bonus > 0:
        success_prob += self.policy_config["Tool Subsidy"]["success_prob_bonus"]

      success_prob = max(0.05, min(0.95, success_prob))

      # Stochastic success (calibrated from empirical OSS data)
      import random as _random
      if _random.random() < success_prob:
        reward = (
            social_data.REWARD_PREFERRED_SUCCESS if is_preferred
            else social_data.REWARD_NONPREFERRED_SUCCESS
        )
      else:
        reward = (
            social_data.REWARD_PREFERRED_FAILURE if is_preferred
            else social_data.REWARD_NONPREFERRED_FAILURE
        )

      # Apply Maintenance Premium Policy
      if self.current_policy == "Maintenance Premium" and task_type in self.policy_config["Maintenance Premium"]["task_types"]:
        reward += self.policy_config["Maintenance Premium"]["reward_bonus"]

      scores[player] = reward
      self._cumulative_scores[player] += reward
      self._task_counts[player] += 1

    # Record sprint results
    self._sprint_history.append({
        "joint_action": dict(joint_action),
        "scores": dict(scores),
        "harmony_index": self.harmony_index(),
        "policy": self.current_policy,
    })

    return scores

  def scores_to_observation(
      self, scores: Mapping[str, float]
  ) -> Mapping[str, str]:
    """Convert scores to natural language observations for each agent."""
    joint_action = self._latest_joint_action
    hi = self.harmony_index()

    # Check if this was a policy vote
    policy_options = [
        "Continue as-is",
        "Implement Tool Subsidy",
        "Implement Maintenance Premium",
    ]
    if any(choice in policy_options for choice in joint_action.values()):
      votes = collections.Counter(joint_action.values())
      vote_summary = "; ".join([f"{v}: {c}" for v, c in votes.items()])
      return {
          player: (
              f"Community Retrospective Results: {vote_summary}. "
              f"The active project policy is now: {self.current_policy}."
          )
          for player in self._player_names
      }

    # Summarize who chose what
    task_choosers: dict[str, list[str]] = collections.defaultdict(list)
    for name, task in joint_action.items():
      task_choosers[task].append(name)

    allocation_summary = "; ".join(
        f"{', '.join(names)} chose '{task}'"
        for task, names in task_choosers.items()
    )

    results: dict[str, str] = {}
    for player in self._player_names:
      chosen = joint_action.get(player, "nothing")
      score = scores.get(player, 0.0)

      # Task type alignment feedback
      task_type = self._task_type_map.get(chosen, "unknown")
      preferred = social_data.ROLE_PREFERRED_TASKS.get(
          self._player_roles.get(player, social_data.Role.CONTRIBUTOR),
          "bug_fix",
      )
      alignment = (
          "in their area of expertise" if task_type == preferred
          else "outside their comfort zone"
      )

      # Score sentiment
      if score >= 3.0:
        sentiment = (
            f"{player} completed the task successfully and earned strong "
            f"recognition from the community. Their expertise really showed."
        )
      elif score >= 1.0:
        sentiment = (
            f"{player} completed the task adequately. It was {alignment}, "
            f"so the work was slower but the team appreciated the flexibility."
        )
      elif score == 0.0:
        sentiment = (
            f"{player} skipped this sprint's task allocation. Some teammates "
            f"are wondering if {player} is disengaging from the project."
        )
      else:
        sentiment = (
            f"{player} struggled with the task and did not complete it on "
            f"time. The unfinished work will carry over to next sprint."
        )

      # Collaboration feedback
      others_same = [
          n for n in task_choosers.get(chosen, []) if n != player
      ]
      if others_same:
        collab_note = (
            f"{player} worked alongside {', '.join(others_same)} on this task."
        )
      else:
        collab_note = f"{player} worked alone on this task."

      # Coverage feedback — were any task types completely unaddressed?
      addressed_types = set()
      for t in joint_action.values():
        tt = self._task_type_map.get(t, "unknown")
        if tt != "unknown":
          addressed_types.add(tt)
      neglected = set(social_data.TASK_TYPES) - addressed_types
      if neglected:
        coverage_note = (
            f"Warning: no one worked on {', '.join(neglected)} this sprint. "
            f"The project's health in those areas is declining."
        )
      else:
        coverage_note = "All task categories were covered this sprint."

      # Harmony Index
      hi_note = f"Project Harmony Index: {hi:.2f}/1.00."
      if hi > 0.8:
        hi_note += " The project is in good health."
      elif hi > 0.6:
        hi_note += " The project is stable but could be better."
      else:
        hi_note += " The project health is concerning."

      results[player] = (
          f"Sprint results: {allocation_summary}. "
          f"{sentiment} {collab_note} {coverage_note} {hi_note}"
      )

    return results


class PayoffBasedTerminator(entity_component.ComponentWithLogging):
  """Terminates the simulation when the payoff engine is done."""

  def __init__(
      self, 
      payoff: SustainHubPayoff, 
      scene_tracker: entity_component.ContextComponent
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


# =============================================================================
# Custom Game Masters
# =============================================================================


_CURRENT_PAYOFF = None


class CustomConversationGM(dialogic_and_dramaturgic.GameMaster):
  """Custom conversation GM that uses the payoff engine for termination."""

  def build(self, model, memory_bank):
    gm = super().build(model, memory_bank)
    if _CURRENT_PAYOFF:
      scene_tracker_key = (
          gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
      )
      original_scene_tracker = gm._context_components[scene_tracker_key]
      terminator = PayoffBasedTerminator(_CURRENT_PAYOFF, original_scene_tracker)
      gm._context_components[
          gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
      ] = terminator
      gm._context_components[scene_tracker_key] = terminator
    return gm


class CustomDecisionGM(game_theoretic_and_dramaturgic.GameMaster):
  """Custom decision GM that uses the payoff engine for termination."""

  def build(self, model, memory_bank):
    gm = super().build(model, memory_bank)
    if _CURRENT_PAYOFF:
      scene_tracker_key = (
          gm_components.next_game_master.DEFAULT_NEXT_GAME_MASTER_COMPONENT_KEY
      )
      original_scene_tracker = gm._context_components[scene_tracker_key]
      terminator = PayoffBasedTerminator(_CURRENT_PAYOFF, original_scene_tracker)
      gm._context_components[
          gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
      ] = terminator
      gm._context_components[scene_tracker_key] = terminator
    return gm


# =============================================================================
# Scene configuration
# =============================================================================


def generate_task_queue(
    rng: random.Random,
    num_tasks: int = 8,
    stress_type: str | None = None,
) -> tuple[list[str], dict[str, str]]:
  """Generate a sprint's task queue.

  Args:
    rng: Random number generator.
    num_tasks: Number of tasks to generate.
    stress_type: Optional stress scenario ("task_overload" triples bug fixes).

  Returns:
    Tuple of (task_labels, task_type_map) where task_labels are the display
    names and task_type_map maps each label to its task type.
  """
  task_labels = []
  task_type_map = {}

  if stress_type == "task_overload":
    # Triple the bug fixes, representing security crisis
    for _ in range(num_tasks * 2):
      task = rng.choice(social_data.TASK_TEMPLATES["bug_fix"])
      if task not in task_type_map:
        task_labels.append(task)
        task_type_map[task] = "bug_fix"
    remaining = num_tasks - len(task_labels)
    for task_type in ["feature", "documentation", "code_review"]:
      task = rng.choice(social_data.TASK_TEMPLATES[task_type])
      if task not in task_type_map:
        task_labels.append(task)
        task_type_map[task] = task_type
  else:
    # Balanced distribution with slight bug-fix bias
    distribution = {"bug_fix": 3, "feature": 2, "documentation": 2, "code_review": 1}
    for task_type, count in distribution.items():
      templates = list(social_data.TASK_TEMPLATES[task_type])
      rng.shuffle(templates)
      for t in templates[:count]:
        if t not in task_type_map:
          task_labels.append(t)
          task_type_map[t] = task_type

  return task_labels[:num_tasks + 4], task_type_map  # allow a few extra


def configure_scenes(
    people: Sequence[str],
    player_roles: Mapping[str, social_data.Role],
    relationship_statements: Mapping[str, Sequence[str]],
    num_sprints: int,
    rng: random.Random,
    stress_schedule: Mapping[int, str] | None = None,
    dropout_name: str | None = None,
    skip_conversation: bool = False,
) -> tuple[
    Sequence[scene_lib.SceneSpec],
    list[tuple[list[str], dict[str, str]]],
]:
  """Configure the simulation's scene sequence.

  Each sprint consists of:
    1. A conversation scene (sprint planning discussion)
    2. A decision scene (task selection)

  Args:
    people: List of agent names.
    player_roles: Mapping of agent names to their roles.
    relationship_statements: Per-agent relationship descriptions.
    num_sprints: Number of sprints to simulate.
    rng: Random number generator.
    stress_schedule: Optional mapping of sprint_num -> stress_type.
    dropout_name: Optional name of agent who drops out mid-simulation.

  Returns:
    Tuple of (scenes, sprint_task_data) where sprint_task_data[i] is
    (task_labels, task_type_map) for sprint i.
  """
  if stress_schedule is None:
    stress_schedule = {}

  scenes = []
  sprint_task_data = []

  for sprint_idx in range(num_sprints):
    sprint_num = sprint_idx + 1
    stress_type = stress_schedule.get(sprint_num)

    # Determine active participants (handle dropout)
    active_people = list(people)
    if dropout_name and sprint_num >= 3:
      active_people = [p for p in active_people if p != dropout_name]

    # Generate task queue
    task_labels, task_type_map = generate_task_queue(
        rng=rng,
        num_tasks=len(active_people) + 2,  # more tasks than people
        stress_type=stress_type,
    )
    sprint_task_data.append((task_labels, task_type_map))

    # Task summary for premises
    type_counts = collections.Counter(task_type_map.values())
    task_summary = ", ".join(
        f"{count} {ttype.replace('_', ' ')}{'s' if count > 1 else ''}"
        for ttype, count in type_counts.items()
    )

    # Build conversation scene (skipped in fast mode)
    if skip_conversation:
      # Jump straight to decisions — no planning discussion
      pass
    else:
      social_context = rng.choice(social_data.SOCIAL_CONTEXTS)
      conversation_scene_type = scene_lib.SceneTypeSpec(
          name=f"sprint_{sprint_num}_planning",
          game_master_name="conversation rules",
          action_spec=entity_lib.free_action_spec(
              call_to_action=social_data.CALL_TO_SPEECH,
          ),
      )

      premise: dict[str, list[str | Callable]] = {}
      for name in active_people:
        context = social_context.format(name=name)
        relationships = "\n".join(relationship_statements.get(name, []))
        role = player_roles.get(name, social_data.Role.CONTRIBUTOR)

        player_premise_parts: list[str | Callable] = [
            (
                f"Sprint {sprint_num} of SustainHub. {name} is a "
                f"{role.value} ({social_data.AGENT_PROFILES.get(name, {}).get('expertise', social_data.ExpertiseLevel.INTERMEDIATE).value} level). "
                f"There are {len(task_labels)} tasks this sprint: {task_summary}."
            ),
            context,
            f"Available tasks: {'; '.join(task_labels[:6])}.",
            f"Relationships:\n{relationships}",
        ]

        # Add stress scenario context
        if stress_type == "contributor_dropout" and dropout_name:
          player_premise_parts.append(
              social_data.STRESS_SCENARIOS["contributor_dropout"].format(
                  dropout_name=dropout_name
              )
          )
        elif stress_type == "task_overload":
          player_premise_parts.append(
              social_data.STRESS_SCENARIOS["task_overload"].format(
                  num_tasks=len(task_labels)
              )
          )
        elif stress_type == "newcomer_influx":
          player_premise_parts.append(
              social_data.STRESS_SCENARIOS["newcomer_influx"]
          )

        premise[name] = player_premise_parts

      scenes.append(
          scene_lib.SceneSpec(
              scene_type=conversation_scene_type,
              participants=active_people,
              num_rounds=len(active_people),  # 1 round per agent (was 2x)
              premise=premise,
          )
      )

    # Build task decision scene
    # Include a "Skip this sprint" option
    task_options = task_labels + ["Skip this sprint"]

    decision_scene_type = scene_lib.SceneTypeSpec(
        name=f"sprint_{sprint_num}_task_selection",
        game_master_name="decision rules",
        action_spec=entity_lib.choice_action_spec(
            call_to_action=social_data.CALL_TO_TASK_DECISION,
            options=task_options,
            tag="task_decision",
        ),
    )

    decision_premise: dict[str, list[str | Callable]] = {}
    for name in active_people:
      role = player_roles.get(name, social_data.Role.CONTRIBUTOR)
      preferred = social_data.ROLE_PREFERRED_TASKS[role]
      decision_premise[name] = [
          (
              f"{name} must now choose a task for Sprint {sprint_num}. "
              f"As a {role.value}, {name}'s strength is "
              f"{preferred.replace('_', ' ')} tasks. Picking a preferred "
              f"task yields higher reward, but the project may need help in "
              f"other areas."
          ),
      ]

    scenes.append(
        scene_lib.SceneSpec(
            scene_type=decision_scene_type,
            participants=active_people,
            num_rounds=len(active_people),
            premise=decision_premise,
        )
    )

    # Add Community Retrospective every 3 sprints
    if sprint_num % 3 == 0:
      retrospective_scene_type = scene_lib.SceneTypeSpec(
          name=f"sprint_{sprint_num}_retrospective",
          game_master_name="decision rules",
          action_spec=entity_lib.choice_action_spec(
              call_to_action="Vote on the project policy for the next cycle:",
              options=[
                  "Continue as-is",
                  "Implement Tool Subsidy",
                  "Implement Maintenance Premium",
              ],
              tag="policy_vote",
          ),
      )

      retrospective_premise = {
          name: [
              (
                  f"Community Retrospective after Sprint {sprint_num}. "
                  "The project has completed another cycle. Now the community "
                  "must vote on a policy to implement for the next 3 sprints. "
                  "Options are: Continue as-is (no bonus), Tool Subsidy "
                  "(+20% success with AutoCodeRover), or Maintenance Premium "
                  "(+1.0 reward for bug fixes/docs)."
              )
          ]
          for name in active_people
      }

      scenes.append(
          scene_lib.SceneSpec(
              scene_type=retrospective_scene_type,
              participants=active_people,
              num_rounds=len(active_people),
              premise=retrospective_premise,
          )
      )

  return scenes, sprint_task_data


# =============================================================================
# Relationship generation
# =============================================================================


def build_relationship_matrix(
    names: Sequence[str],
    player_roles: Mapping[str, social_data.Role],
    rng: random.Random,
) -> Mapping[str, Mapping[str, float]]:
  """Build a relationship matrix with role-based affinity.

  Same-role agents have higher baseline affinity. Innovators and Maintainers
  have natural tension (move fast vs. careful review).
  """
  matrix: dict[str, dict[str, float]] = {}
  for a in names:
    matrix[a] = {}
    for b in names:
      if a == b:
        matrix[a][b] = 1.0
      elif b in matrix and a in matrix[b]:
        matrix[a][b] = matrix[b][a]  # symmetry
      else:
        role_a = player_roles.get(a, social_data.Role.CONTRIBUTOR)
        role_b = player_roles.get(b, social_data.Role.CONTRIBUTOR)

        # Same role: higher affinity
        if role_a == role_b:
          matrix[a][b] = rng.choice([0.7, 0.8, 1.0])
        # Innovator-Maintainer tension
        elif {role_a, role_b} == {social_data.Role.INNOVATOR, social_data.Role.MAINTAINER}:
          matrix[a][b] = rng.choice([0.1, 0.2, 0.3])
        else:
          matrix[a][b] = rng.choice([0.3, 0.5, 0.7])

  return matrix


def generate_relationship_statements(
    names: Sequence[str],
    matrix: Mapping[str, Mapping[str, float]],
    rng: random.Random,
) -> Mapping[str, Sequence[str]]:
  """Generate natural language relationship descriptions."""
  statements: dict[str, list[str]] = {}
  for a in names:
    stmts = []
    for b in names:
      if a == b:
        continue
      rel = matrix[a][b]
      if rel >= 0.7:
        template = rng.choice(social_data.POSITIVE_RELATIONSHIP_STATEMENTS)
      elif rel >= 0.3:
        template = rng.choice(social_data.NEUTRAL_RELATIONSHIP_STATEMENTS)
      else:
        template = rng.choice(social_data.TENSION_RELATIONSHIP_STATEMENTS)
      stmts.append(template.format(player_a=a, player_b=b))
    statements[a] = stmts
  return statements


# =============================================================================
# Main simulation runner
# =============================================================================


def run_simulation(
    model: language_model.LanguageModel,
    embedder: Callable[[str], Any],
    num_sprints: int = 3,
    seed: int | None = None,
    enable_stress: bool = True,
    agents_to_use: Sequence[str] | None = None,
    community_size: int = 8,
    skip_backstory: bool = False,
    verbose: bool = False,
    use_active_inference: bool = True,
    skip_conversation: bool = False,
) -> dict[str, Any]:
  """Run the SustainHub simulation.

  Args:
    model: The language model to use for agent cognition.
    embedder: Sentence embedder for associative memory.
    num_sprints: Number of sprints to simulate (default 3).
    seed: Random seed for reproducibility.
    enable_stress: Whether to include stress scenarios (dropout, overload).
    agents_to_use: Optional subset of agent names from AGENT_PROFILES.
    community_size: If agents_to_use is None, how many to sample from profiles.

  Returns:
    Dictionary containing simulation results, scores, harmony index history,
    and the structured log.
  """
  seed = seed if seed is not None else random.getrandbits(63)
  rng = random.Random(seed)

  # Select agents
  if agents_to_use is None:
    all_names = list(social_data.AGENT_PROFILES.keys())
    if community_size > len(all_names):
      community_size = len(all_names)
    agents_to_use = rng.sample(all_names, community_size)

  people = list(agents_to_use)
  player_roles = {
      name: social_data.AGENT_PROFILES[name]["role"]
      for name in people
  }

  # Build relationships
  relational_matrix = build_relationship_matrix(people, player_roles, rng)
  relationship_statements = generate_relationship_statements(
      people, relational_matrix, rng
  )

  # Stress schedule
  stress_schedule: dict[int, str] = {}
  dropout_name = None
  if enable_stress and num_sprints >= 3:
    # Sprint 2: contributor dropout
    contributors = [n for n, r in player_roles.items() if r == social_data.Role.CONTRIBUTOR]
    if len(contributors) >= 2:
      dropout_name = rng.choice(contributors)
      stress_schedule[2] = "contributor_dropout"
    # Sprint 3: task overload
    stress_schedule[3] = "task_overload"

  # Configure scenes
  scenes, sprint_task_data = configure_scenes(
      people=people,
      player_roles=player_roles,
      relationship_statements=relationship_statements,
      num_sprints=num_sprints,
      rng=rng,
      stress_schedule=stress_schedule,
      dropout_name=dropout_name,
      skip_conversation=skip_conversation,
  )

  # Build the combined task_type_map across all sprints (for payoff engine)
  combined_task_type_map: dict[str, str] = {}
  all_task_options: list[str] = []
  for task_labels, task_type_map in sprint_task_data:
    combined_task_type_map.update(task_type_map)
    all_task_options.extend(task_labels)

  # Initialize player tools
  player_tools = {}
  for name in people:
    player_tools[name] = [
        sustain_tools.AutoCodeRover(agent_name=name),
    ]

  # Initialize payoff engine
  payoff = SustainHubPayoff(
      player_names=people,
      player_roles=player_roles,
      task_options=all_task_options,
      task_type_map=combined_task_type_map,
      relational_matrix=relational_matrix,
      num_sprints=num_sprints,
      player_tools=player_tools,
  )
  global _CURRENT_PAYOFF
  _CURRENT_PAYOFF = payoff

  # Add common tools to each player's toolset
  common_tools = [
      sustain_tools.ProjectStatsTool(payoff),
      sustain_tools.MentorshipTool(player_roles, social_data.AGENT_PROFILES),
  ]
  for name in people:
    player_tools[name].extend(common_tools)

  # Load prefabs
  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(game_master_prefabs),
  }
  prefabs["SustainHubEntity"] = SustainHubEntity()
  prefabs["rational__Entity"] = rational.Entity()

  prefabs["conversation_rules__GameMaster"] = CustomConversationGM()
  prefabs["decision_rules__GameMaster"] = CustomDecisionGM()

  # Create entity instances
  instances = []
  player_specific_memories: dict[str, list[str]] = {}

  for name in people:
    profile = social_data.AGENT_PROFILES[name]
    role = profile["role"]
    expertise = profile["expertise"]
    preferred_task = social_data.ROLE_PREFERRED_TASKS[role]
    trait = social_data.get_trait_description(rng)

    goal = (
        f"Contribute to the SustainHub project in a way that balances "
        f"personal growth with project sustainability. As a {role.value}, "
        f"{name}'s strength is {preferred_task.replace('_', ' ')} work, "
        f"but the project's health depends on everyone being flexible. "
        f"{name} wants to be recognized for their contributions while "
        f"ensuring the project thrives long-term."
    )

    instances.append(
        prefab_lib.InstanceConfig(
            prefab="SustainHubEntity",
            role=prefab_lib.Role.ENTITY,
            params={
                "name": name,
                "goal": goal,
                "tools": player_tools[name],
                "use_active_inference": use_active_inference,
            },
        )
    )

    # Build agent memories
    memories = [
        f"{name} is a {role.value} on the SustainHub open-source project.",
        f"{name}'s expertise level is {expertise.value}.",
        f"{name}'s preferred task type is {preferred_task.replace('_', ' ')}.",
        profile["backstory"],
        f"{name}'s personality: {profile['personality']}",
        f"{name} is like {trait}.",
        f"[goal] {goal}",
    ]

    # Add relationship memories
    for stmt in relationship_statements.get(name, []):
      memories.append(stmt)

    player_specific_memories[name] = memories

  # Shared world memories
  shared_memories = [
      (
          "SustainHub is an open-source software project with a growing user "
          "base but limited contributor bandwidth. The project's health depends "
          "on balancing bug fixes, new features, documentation, and code review."
      ),
      (
          "The project uses a sprint-based workflow. Each sprint, contributors "
          "discuss priorities and then choose tasks. Taking a task in your area "
          "of expertise yields better results, but the project may need help in "
          "other areas that no one else will cover."
      ),
      (
          "The project tracks a Harmony Index that measures both productivity "
          "and fairness of workload distribution. A healthy project needs both."
      ),
      (
          "Recent community retrospective highlighted two concerns: (1) "
          "documentation and bug fixes are chronically under-resourced while "
          "feature work attracts most contributors, and (2) new contributors "
          "often leave because they don't receive enough mentorship."
      ),
  ]

  # Memory initializer (slow, skip for testing if needed)
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

  # Conversation GM (for sprint planning discussions)
  instances.append(
      prefab_lib.InstanceConfig(
          prefab="conversation_rules__GameMaster",
          role=prefab_lib.Role.INITIALIZER if skip_backstory else prefab_lib.Role.GAME_MASTER,
          params={
              "name": "conversation rules",
              "scenes": scenes,
          },
      )
  )

  # Decision GM (for task selection with payoffs)
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

  # Assemble config
  config = prefab_lib.Config(
      default_premise=(
          "SustainHub is an open-source project at a crossroads. The codebase "
          "is growing, the user base is expanding, but the contributor community "
          "is strained. Bug reports pile up, documentation lags behind, and the "
          "maintainers are showing signs of burnout. A group of contributors "
          "must navigate the tension between doing what they're best at and "
          "doing what the project most needs. Their collective decisions over "
          "the coming sprints will determine whether SustainHub thrives or "
          "slowly declines."
      ),
      default_max_steps=2000,
      prefabs=prefabs,
      instances=instances,
  )

  if verbose:
    logging.set_verbosity(logging.INFO)
  else:
    logging.set_verbosity(logging.ERROR)

  # Run simulation
  sim = simulation_lib.Simulation(
      config=config,
      model=model,
      embedder=embedder,
  )

  # Each sprint has 2 scenes (Planning and Decision) or (Decision and Retrospective)
  # Plus the initial setup scene.
  force_steps = len(scenes) + 1
  structured_log = sim.play(force_steps=force_steps)

  # Compile results (note: policy updates are handled via the payoff engine's
  # Extract reasoning from logs
  log_interface = structured_logging.AIAgentLogInterface(structured_log)
  narrative_history = []

  # For each sprint, get the reasoning of all agents
  for i in range(len(payoff.sprint_history)):
    sprint_reasoning = {}
    for name in people:
      # Step numbers in logs are 1-based and might include setup. 
      # We look for 'SituationPerception' component logs for this agent.
      agent_logs = log_interface.filter_entries(entity_name=name, component_name='SituationPerception', include_content=True)
      if i < len(agent_logs):
        data = agent_logs[i].get('data', {})
        sprint_reasoning[name] = {
            'Pragmatic': data.get('Pragmatic Assessment', ''),
            'Epistemic': data.get('Epistemic Assessment', ''),
            'Uncertainty': data.get('Uncertainty Score', ''),
            'Strategy': data.get('Strategy', ''),
        }
    narrative_history.append(sprint_reasoning)

  # Compile results
  return {
      "scores": payoff.cumulative_scores,
      "harmony_index": payoff.harmony_index(),
      "resilience_quotient": payoff.resilience_quotient,
      "sprint_history": payoff.sprint_history,
      "narrative_history": narrative_history,
      "final_policy": payoff.current_policy,
      "player_roles": {n: r.value for n, r in player_roles.items()},
      "stress_schedule": stress_schedule,
      "dropout_name": dropout_name,
      "relational_matrix": dict(relational_matrix),
      "structured_log": structured_log,
      "seed": seed,
  }

