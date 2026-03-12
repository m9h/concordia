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

"""Tools for agents in the SustainHub simulation."""

from typing import Any, Mapping, Sequence
from concordia.document import tool as tool_lib
from examples.games.sustain_hub import social_data


class AutoCodeRover(tool_lib.Tool):
  """Tool for autonomous program improvement and patch generation."""

  def __init__(self, agent_name: str):
    self._agent_name = agent_name
    self.last_used_sprint = -1
    self.usage_count = 0

  @property
  def name(self) -> str:
    return "use_autocode_rover"

  @property
  def description(self) -> str:
    return "Uses an autonomous AI agent to analyze the codebase and generate a patch for a bug or feature. Highly recommended for 'hard' tasks. Args: task_description (str)"

  def execute(self, **kwargs: Any) -> str:
    task = kwargs.get("task_description", "the current task")
    self.usage_count += 1
    # We'll rely on the simulation to check this usage_count or similar
    return f"AutoCodeRover: Analyzed {task}. Identified relevant context in 4 files. Generated a candidate patch with 85% confidence."


class ProjectStatsTool(tool_lib.Tool):
  """Tool to check the current health and neglected areas of the project."""

  def __init__(self, payoff_engine: Any):
    self._payoff = payoff_engine

  @property
  def name(self) -> str:
    return "check_project_stats"

  @property
  def description(self) -> str:
    return "Returns the current Harmony Index and lists task types that were neglected in the previous sprint. Args: none"

  def execute(self, **kwargs: Any) -> str:
    del kwargs
    hi = self._payoff.harmony_index()
    history = self._payoff.sprint_history
    if not history:
      return f"Project is at start. Current Harmony Index: {hi:.2f}. No history yet."

    last_sprint = history[-1]
    joint_action = last_sprint["joint_action"]

    addressed_types = set()
    for task in joint_action.values():
      if not task: continue
      # This is a bit hacky as we don't have the task_type_map easily accessible here
      # but we can try to guess from the task name or just report what we have.
      # For better implementation, we should pass more info to the tool.
      pass

    return f"Current Harmony Index: {hi:.2f}. (1.00 is perfect, <0.60 is concerning)."


class MentorshipTool(tool_lib.Tool):
  """Tool to identify team members who may need mentorship."""

  def __init__(self, player_roles: Mapping[str, social_data.Role], agent_profiles: Mapping[str, Any]):
    self._player_roles = player_roles
    self._agent_profiles = agent_profiles

  @property
  def name(self) -> str:
    return "check_mentorship_needs"

  @property
  def description(self) -> str:
    return "Lists all current project members and their expertise levels to identify who might need help. Args: none"

  def execute(self, **kwargs: Any) -> str:
    del kwargs
    lines = ["Project Members Expertise:"]
    for name, profile in self._agent_profiles.items():
      expertise = profile["expertise"].value
      role = profile["role"].value
      lines.append(f"- {name}: {role} ({expertise} level)")
    return "\n".join(lines)
