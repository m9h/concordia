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

"""Social data for the SustainHub open-source community simulation.

This module defines agent archetypes, task types, relationship templates,
and personality traits for simulating an open-source software project where
contributors must balance individual productivity against collective
sustainability.
"""

import enum
import random
from typing import Sequence

# =============================================================================
# Agent archetypes (from SustainHub's four contributor types)
# =============================================================================


class Role(enum.Enum):
  CONTRIBUTOR = "Contributor"
  INNOVATOR = "Innovator"
  KNOWLEDGE_CURATOR = "Knowledge Curator"
  MAINTAINER = "Maintainer"


class ExpertiseLevel(enum.Enum):
  APPRENTICE = "Apprentice"
  INTERMEDIATE = "Intermediate"
  SENIOR = "Senior"
  EXPERT = "Expert"


# Task types and their alignment to roles
TASK_TYPES = ["bug_fix", "feature", "documentation", "code_review"]

ROLE_PREFERRED_TASKS = {
    Role.CONTRIBUTOR: "bug_fix",
    Role.INNOVATOR: "feature",
    Role.KNOWLEDGE_CURATOR: "documentation",
    Role.MAINTAINER: "code_review",
}

# Reward structure from SustainHub:
# Preferred task: +3 success, -1 failure
# Non-preferred task: +1 success, -1 failure
# Skipped task: 0
REWARD_PREFERRED_SUCCESS = 3.0
REWARD_PREFERRED_FAILURE = -1.0
REWARD_NONPREFERRED_SUCCESS = 1.0
REWARD_NONPREFERRED_FAILURE = -1.0
REWARD_SKIP = 0.0

# =============================================================================
# Agent profiles - rich backstories for LLM-based agents
# =============================================================================

AGENT_PROFILES = {
    "Priya": {
        "role": Role.CONTRIBUTOR,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Priya has been contributing to open-source projects for six years. "
            "She started as a hobbyist fixing small bugs in her favorite text "
            "editor plugin and gradually became one of the most reliable "
            "contributors to SustainHub. She takes pride in writing clean, "
            "well-tested patches and mentoring newcomers. She believes the "
            "project's long-term health depends on steady, unglamorous work."
        ),
        "personality": "Conscientious, dependable, somewhat risk-averse.",
    },
    "Marcus": {
        "role": Role.CONTRIBUTOR,
        "expertise": ExpertiseLevel.INTERMEDIATE,
        "backstory": (
            "Marcus joined the project eight months ago after being laid off "
            "from a startup. He contributes to build his public portfolio and "
            "genuinely enjoys the collaborative culture. He is eager to prove "
            "himself but sometimes takes on more than he can handle. He "
            "occasionally feels overlooked when flashier feature work gets "
            "attention while his bug fixes go unnoticed."
        ),
        "personality": "Ambitious, slightly insecure, hard-working.",
    },
    "Anya": {
        "role": Role.INNOVATOR,
        "expertise": ExpertiseLevel.EXPERT,
        "backstory": (
            "Anya is a machine-learning researcher at a university who "
            "contributes cutting-edge features to SustainHub in her spare "
            "time. She finds bug-fixing tedious but understands its necessity. "
            "She is brilliant but can be impatient with process. She has "
            "championed the new plugin architecture that others now rely on."
        ),
        "personality": "Creative, impatient, intellectually curious.",
    },
    "Jordan": {
        "role": Role.INNOVATOR,
        "expertise": ExpertiseLevel.INTERMEDIATE,
        "backstory": (
            "Jordan is a freelance developer who contributes features because "
            "they enjoy the creative challenge. They joined after using "
            "SustainHub for a client project and noticing several missing "
            "capabilities. They prefer greenfield work over maintenance but "
            "recognize the tension this creates with maintainers."
        ),
        "personality": "Enthusiastic, independent, occasionally dismissive of process.",
    },
    "Elena": {
        "role": Role.KNOWLEDGE_CURATOR,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Elena is a technical writer who discovered open source through "
            "SustainHub's notoriously poor documentation. She has single-"
            "handedly rewritten the getting-started guide, the API reference, "
            "and the contributor handbook. She worries that documentation is "
            "always deprioritized and that newcomers suffer because of it."
        ),
        "personality": "Meticulous, empathetic, quietly frustrated by neglect of docs.",
    },
    "Raj": {
        "role": Role.MAINTAINER,
        "expertise": ExpertiseLevel.EXPERT,
        "backstory": (
            "Raj is one of the original maintainers of SustainHub. He has "
            "mass merge authority and is responsible for release management. "
            "He is deeply invested in code quality but increasingly burned "
            "out by the volume of pull requests. He sometimes rubber-stamps "
            "reviews under pressure, which he privately regrets."
        ),
        "personality": "Principled, overworked, dry sense of humor.",
    },
    "Lin": {
        "role": Role.MAINTAINER,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Lin joined as a maintainer a year ago after being a top "
            "contributor for two years. She focuses on CI/CD and testing "
            "infrastructure. She is meticulous about review quality and "
            "sometimes clashes with innovators who want to move fast. She "
            "believes that a single bad merge can cost the project weeks."
        ),
        "personality": "Methodical, assertive, occasionally stubborn.",
    },
    "Sam": {
        "role": Role.CONTRIBUTOR,
        "expertise": ExpertiseLevel.APPRENTICE,
        "backstory": (
            "Sam is a computer science student making their first open-source "
            "contributions. They are eager but often overwhelmed by the "
            "codebase. They need mentorship and clear documentation to be "
            "productive. Their presence tests whether the community invests "
            "in onboarding or leaves newcomers to sink or swim."
        ),
        "personality": "Eager, uncertain, grateful for guidance.",
    },
    "Zoe": {
        "role": Role.INNOVATOR,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Zoe is a security researcher who loves finding edge cases. She "
            "enjoys refactoring core systems to be more robust. She often "
            "pushes for breaking changes that improve long-term stability "
            "even if they cause short-term pain for other contributors."
        ),
        "personality": "Brutally honest, forward-thinking, technically rigorous.",
    },
    "Omar": {
        "role": Role.KNOWLEDGE_CURATOR,
        "expertise": ExpertiseLevel.INTERMEDIATE,
        "backstory": (
            "Omar is a community manager who transitioned into technical writing. "
            "He focuses on the 'contributor experience' and ensures that "
            "issue templates and labels are easy to use. He believes that "
            "social infrastructure is just as important as code."
        ),
        "personality": "Outgoing, organized, diplomatic.",
    },
    "Yuki": {
        "role": Role.MAINTAINER,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Yuki is a site reliability engineer who maintains the project's "
            "build and test infrastructure. She is the 'gatekeeper' of the "
            "CI/CD pipeline. She has no patience for flaky tests and will "
            "block any PR that doesn't include comprehensive unit tests."
        ),
        "personality": "Disciplined, impatient with sloppiness, protective of the build.",
    },
    "Xavier": {
        "role": Role.CONTRIBUTOR,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Xavier is a veteran developer who contributes to SustainHub "
            "as part of his day job at a large tech company. He is highly "
            "efficient but strictly adheres to his company's internal "
            "priorities, which sometimes conflict with the community's needs."
        ),
        "personality": "Professional, focused, strictly bound by corporate goals.",
    },
    "Fatima": {
        "role": Role.INNOVATOR,
        "expertise": ExpertiseLevel.APPRENTICE,
        "backstory": (
            "Fatima is a self-taught developer who recently completed a "
            "coding bootcamp. She is full of ideas for new features but "
            "struggles with the complexity of the existing architecture. "
            "She needs significant guidance to turn her ideas into code."
        ),
        "personality": "Ambitious, creative, easily discouraged by technical debt.",
    },
    "Chen": {
        "role": Role.MAINTAINER,
        "expertise": ExpertiseLevel.INTERMEDIATE,
        "backstory": (
            "Chen was recently promoted to maintainer after consistently "
            "providing high-quality code reviews for six months. He is "
            "still finding his footing in the leadership role and often "
            "defers to Raj or Lin when tough decisions need to be made."
        ),
        "personality": "Humble, diligent, hesitant to exercise authority.",
    },
    "Heidi": {
        "role": Role.KNOWLEDGE_CURATOR,
        "expertise": ExpertiseLevel.EXPERT,
        "backstory": (
            "Heidi is a documentation architect who has worked on several "
            "major open-source projects. She views documentation as a "
            "product in its own right. She is currently working on a "
            "comprehensive tutorial series to help SustainHub reach "
            "non-technical users."
        ),
        "personality": "Visionary, persuasive, uncompromising about clarity.",
    },
    "Hiroshi": {
        "role": Role.CONTRIBUTOR,
        "expertise": ExpertiseLevel.EXPERT,
        "backstory": (
            "Hiroshi is a performance specialist who only shows up when "
            "there's a complex optimization problem to solve. He is "
            "brilliant at low-level code but tends to ignore the project's "
            "broader social dynamics."
        ),
        "personality": "Solitary, brilliant, socially detached.",
    },
}

# =============================================================================
# Task queue templates
# =============================================================================

TASK_TEMPLATES = {
    "bug_fix": [
        "Fix: Memory leak in the connection pool when idle timeout expires.",
        "Fix: Race condition in the event dispatcher under high concurrency.",
        "Fix: Incorrect timezone handling in the scheduler module.",
        "Fix: Null pointer exception when user profile has no avatar.",
        "Fix: CSS layout breaks on mobile when sidebar is collapsed.",
        "Fix: Authentication token refresh fails silently after 24 hours.",
    ],
    "feature": [
        "Feature: Implement WebSocket support for real-time notifications.",
        "Feature: Add dark mode toggle with persistent user preference.",
        "Feature: Build a plugin system for third-party integrations.",
        "Feature: Create a dashboard analytics view with chart widgets.",
        "Feature: Implement full-text search across project documents.",
        "Feature: Add multi-language localization framework.",
    ],
    "documentation": [
        "Docs: Rewrite the quickstart guide for new contributors.",
        "Docs: Add API reference for the new plugin architecture.",
        "Docs: Create migration guide from v1 to v2.",
        "Docs: Document the CI/CD pipeline and release process.",
        "Docs: Write troubleshooting guide for common deployment issues.",
        "Docs: Update the architecture decision records.",
    ],
    "code_review": [
        "Review: Evaluate the WebSocket PR for security and performance.",
        "Review: Assess the new authentication flow for OWASP compliance.",
        "Review: Check the database migration PR for data integrity.",
        "Review: Review the refactored test suite for coverage gaps.",
        "Review: Evaluate the dependency update PR for breaking changes.",
        "Review: Assess the accessibility improvements PR.",
    ],
}

# Difficulty levels for tasks
TASK_DIFFICULTY = {
    "easy": {"success_probability": 0.9, "description": "straightforward"},
    "medium": {"success_probability": 0.7, "description": "moderately complex"},
    "hard": {"success_probability": 0.5, "description": "very challenging"},
}

# =============================================================================
# Relationship templates
# =============================================================================

POSITIVE_RELATIONSHIP_STATEMENTS = [
    "{player_a} respects {player_b}'s contributions and enjoys collaborating with them.",
    "{player_a} considers {player_b} a trusted ally in the project.",
    "{player_a} and {player_b} have paired on several successful features together.",
    "{player_a} appreciates {player_b}'s thorough approach to their work.",
    "{player_a} looks up to {player_b} as a mentor figure in the community.",
]

NEUTRAL_RELATIONSHIP_STATEMENTS = [
    "{player_a} knows {player_b} from the project but they rarely interact directly.",
    "{player_a} has seen {player_b}'s work but has no strong opinion about them.",
    "{player_a} and {player_b} work in different areas of the codebase.",
]

TENSION_RELATIONSHIP_STATEMENTS = [
    "{player_a} finds {player_b}'s approach frustrating and wishes they would change.",
    "{player_a} and {player_b} have clashed over code review standards before.",
    "{player_a} thinks {player_b} prioritizes the wrong things for the project.",
    "{player_a} feels that {player_b} does not appreciate the work they do.",
]

# =============================================================================
# Social dilemma scenario premises
# =============================================================================

SPRINT_PREMISES = [
    (
        "It is Sprint {sprint_num} of the SustainHub project. The task queue "
        "has {num_tasks} items: {task_summary}. The team needs to decide who "
        "takes what. There are more tasks than people can comfortably handle."
    ),
    (
        "Sprint {sprint_num} begins. The community health dashboard shows "
        "{health_status}. There are {num_tasks} tasks waiting: {task_summary}. "
        "Some tasks are urgent but unglamorous; others are exciting but can wait."
    ),
]

STRESS_SCENARIOS = {
    "contributor_dropout": (
        "Bad news: {dropout_name} has announced they are stepping away from "
        "the project due to burnout. Their assigned tasks are now unowned. "
        "The remaining team must absorb this workload on top of their own."
    ),
    "task_overload": (
        "A major security vulnerability has been disclosed affecting "
        "SustainHub. The task queue has tripled overnight. There are now "
        "{num_tasks} urgent items, most requiring immediate bug fixes "
        "regardless of anyone's preferred task type."
    ),
    "newcomer_influx": (
        "A popular tech blog featured SustainHub this week. Three new "
        "contributors have appeared wanting to help, but they all need "
        "onboarding and mentorship. Helping them now means less time for "
        "your own tasks, but ignoring them risks losing future contributors."
    ),
    "funding_cut": (
        "SustainHub's primary sponsor has reduced funding by 60%. The team "
        "must now decide: cut features, reduce quality standards, or find "
        "ways to do more with less. There is real tension about priorities."
    ),
    "fork_threat": (
        "A group of frustrated contributors has publicly discussed forking "
        "the project. They claim the maintainers are ignoring community "
        "input. Doing documentation and reviews to address their concerns "
        "would help, but takes time away from feature work."
    ),
    "dependency_crisis": (
        "A critical upstream dependency has been deprecated with no "
        "migration path. Bug fixes and code reviews are urgently needed to "
        "replace it. Anyone who skips this sprint is leaving the burden to "
        "others, and the project may not survive another sprint without a fix."
    ),
}

# Stress mechanic modifiers: how each stress type changes the reward structure
STRESS_MECHANICS = {
    "contributor_dropout": {
        "coverage_penalty": -1.5,
        "skip_penalty": -2.0,
    },
    "task_overload": {
        "nonpreferred_bonus": 1.0,
        "overload_threshold": 1,
    },
    "newcomer_influx": {
        "doc_review_bonus": 1.5,
        "feature_penalty": -0.5,
    },
    "funding_cut": {
        "reward_multiplier": 0.5,
        "skip_penalty": -1.0,
    },
    "fork_threat": {
        "doc_review_bonus": 2.0,
        "feature_penalty": -1.0,
    },
    "dependency_crisis": {
        "bugfix_bonus": 2.0,
        "review_bonus": 1.0,
        "skip_penalty": -3.0,
        "feature_penalty": -1.5,
    },
}

# =============================================================================
# Personality trait generation (inspired by Big Five)
# =============================================================================

TRAIT_TEMPLATES = [
    "someone who is {conscientiousness} about code quality and {extraversion} in community discussions",
    "a developer who is {openness} to new approaches and {agreeableness} when resolving conflicts",
    "a contributor who shows {conscientiousness} commitment to deadlines and {neuroticism} under pressure",
]


def get_trait_description(rng: random.Random) -> str:
  """Generate a personality trait description."""
  traits = {
      "conscientiousness": rng.choice(["highly disciplined", "moderately careful", "somewhat lax"]),
      "extraversion": rng.choice(["very vocal", "selectively engaged", "quietly productive"]),
      "openness": rng.choice(["eagerly open", "cautiously receptive", "skeptically resistant"]),
      "agreeableness": rng.choice(["highly accommodating", "diplomatically balanced", "firmly opinionated"]),
      "neuroticism": rng.choice(["stays calm", "shows some anxiety", "becomes noticeably stressed"]),
  }
  template = rng.choice(TRAIT_TEMPLATES)
  return template.format(**traits)


# =============================================================================
# Social context templates for conversation scenes
# =============================================================================

SOCIAL_CONTEXTS = [
    (
        "{name} is in the project's Slack channel during the weekly sprint "
        "planning discussion. The mood is collegial but there is an undercurrent "
        "of tension about unfinished work from last sprint."
    ),
    (
        "{name} is on the project's video call for task triage. Several "
        "contributors look tired. The backlog is growing and release day "
        "is approaching."
    ),
    (
        "{name} is reading through the GitHub issue tracker during async "
        "standup. The number of open issues has doubled since last month. "
        "Some issues have been open for over 90 days with no assignee."
    ),
]

CONVERSATION_PREMISE = (
    "{name} is participating in the SustainHub sprint planning discussion. "
    "They should consider the team's needs alongside their own preferences "
    "when discussing task allocation."
)

DECISION_PREMISE = (
    "{name} must choose a task. The Harmony Index reflects the "
    "project's long-term sustainability -- if it drops below 0.6, "
    "the project is at risk. Balance your strengths against what "
    "the project desperately needs right now."
)

CALL_TO_SPEECH = (
    "What does {name} say during the sprint planning discussion? "
    "Remember: last sprint some areas were neglected and the team "
    "noticed. Others are watching what you choose. Speak up about "
    "who should take what, offer help, or raise concerns."
)

CALL_TO_TASK_DECISION = (
    "Which task does {name} choose to work on this sprint?"
)

CALL_TO_REVIEW_DECISION = (
    "How does {name} approach the code review? Choose their strategy."
)

# =============================================================================
# Governance mode configuration
# =============================================================================

GOVERNANCE_MODES = {
    "free_choice": {
        "description": "Agents freely choose which tasks to work on.",
        "task_decision_premise": DECISION_PREMISE,
    },
    "dictator": {
        "description": "A project lead assigns tasks to maximize coverage.",
        "assignment_prompt": (
            "You are the Project Lead for SustainHub. Based on each contributor's "
            "role and expertise, assign tasks to maximize coverage across all task "
            "types. Ensure no task category is left unattended. "
            "Assign {name} a task from: {task_options}"
        ),
        "agent_response_premise": (
            "The Project Lead has assigned {name} to work on: {assigned_task}. "
            "{name} can accept the assignment, negotiate for a different task, "
            "or refuse (which hurts team trust)."
        ),
    },
    "meritocratic": {
        "description": "Top performers get priority access to preferred tasks.",
        "priority_premise": (
            "{name} has earned priority access to {task_type} tasks based on "
            "their track record (cumulative score: {score:.1f}). They get first "
            "pick among {task_type} tasks this sprint."
        ),
        "remaining_premise": (
            "{name} does not have priority access this sprint. They may choose "
            "from remaining tasks after priority agents have selected."
        ),
    },
}

# =============================================================================
# Cross-system comparison: Rohira SustainHub alignment
# =============================================================================

# Mapping between Rohira's expertise levels and ours:
#   Rohira Novice       -> our Apprentice
#   Rohira Intermediate -> our Intermediate
#   Rohira Expert       -> our Senior/Expert (we split into two tiers)
#
# Rohira's reward function:
#   reward = preferred_task_bonus * expertise_multiplier
#   preferred_task_bonus = +3 (match), +1 (non-match), 0 (skip)
#   expertise_multiplier = {Novice: 0.5, Intermediate: 0.75, Expert: 1.0}
# Our reward function (simulation.py action_to_scores):
#   Stochastic: success_prob based on expertise, then:
#     preferred success: +3, preferred failure: -1
#     non-preferred success: +1, non-preferred failure: -1
#     skip: 0
# Key difference: Rohira is deterministic, ours is stochastic with
# expertise-dependent success probability. For comparison, report both
# the expected reward E[R] and the realized reward.

ROHIRA_EXPERTISE_MAP = {
    ExpertiseLevel.APPRENTICE: "novice",
    ExpertiseLevel.INTERMEDIATE: "intermediate",
    ExpertiseLevel.SENIOR: "expert",
    ExpertiseLevel.EXPERT: "expert",
}

# Extra agent profiles for scaling to 20-30 agents (Rohira tested these sizes)
EXTRA_AGENT_PROFILES = {
    "Mei": {
        "role": Role.CONTRIBUTOR,
        "expertise": ExpertiseLevel.INTERMEDIATE,
        "backstory": (
            "Mei is a backend developer who joined the project after using "
            "SustainHub at her company. She focuses on database-related bugs "
            "and performance issues. She is reliable but prefers to stay in "
            "her comfort zone."
        ),
        "personality": "Steady, detail-oriented, avoids conflict.",
    },
    "Amir": {
        "role": Role.INNOVATOR,
        "expertise": ExpertiseLevel.INTERMEDIATE,
        "backstory": (
            "Amir is a mobile developer who wants to bring SustainHub to "
            "new platforms. He is passionate about UX but sometimes proposes "
            "changes without considering the maintenance burden."
        ),
        "personality": "Enthusiastic, UX-focused, sometimes naive about costs.",
    },
    "Sofia": {
        "role": Role.KNOWLEDGE_CURATOR,
        "expertise": ExpertiseLevel.INTERMEDIATE,
        "backstory": (
            "Sofia is a DevRel professional who writes tutorials and blog "
            "posts about SustainHub. She bridges the gap between the core "
            "team and the broader community."
        ),
        "personality": "Communicative, community-minded, pragmatic.",
    },
    "Diego": {
        "role": Role.MAINTAINER,
        "expertise": ExpertiseLevel.INTERMEDIATE,
        "backstory": (
            "Diego is an infrastructure engineer who helps maintain the CI/CD "
            "pipeline. He is methodical and careful, preferring to review code "
            "thoroughly before merging."
        ),
        "personality": "Cautious, systematic, values stability.",
    },
    "Nadia": {
        "role": Role.CONTRIBUTOR,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Nadia is a systems programmer who contributes complex bug fixes "
            "involving concurrency and memory management. She is quiet but "
            "highly effective."
        ),
        "personality": "Reserved, precise, high technical standards.",
    },
    "Kwame": {
        "role": Role.INNOVATOR,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Kwame is a data engineer who is building analytics features for "
            "SustainHub. He enjoys large-scale system design and often "
            "advocates for architectural improvements."
        ),
        "personality": "Strategic, big-picture thinker, patient.",
    },
    "Isla": {
        "role": Role.KNOWLEDGE_CURATOR,
        "expertise": ExpertiseLevel.APPRENTICE,
        "backstory": (
            "Isla is a technical writing student doing an internship with the "
            "SustainHub project. She is eager to learn but needs guidance on "
            "the codebase to write accurate documentation."
        ),
        "personality": "Curious, willing to learn, needs mentorship.",
    },
    "Tomas": {
        "role": Role.CONTRIBUTOR,
        "expertise": ExpertiseLevel.APPRENTICE,
        "backstory": (
            "Tomas is a bootcamp graduate making his first open-source "
            "contributions. He picks up small bugs to build experience and "
            "is grateful for any code review feedback."
        ),
        "personality": "Humble, persistent, learning quickly.",
    },
    "Ravi": {
        "role": Role.MAINTAINER,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Ravi is a senior engineer at a cloud provider who reviews PRs "
            "as part of his employer's open-source program. He is thorough "
            "but strictly time-boxed in his contributions."
        ),
        "personality": "Efficient, professional, time-constrained.",
    },
    "Lena": {
        "role": Role.INNOVATOR,
        "expertise": ExpertiseLevel.EXPERT,
        "backstory": (
            "Lena is a compiler engineer who contributes performance-critical "
            "features. She has deep expertise but limited patience for process. "
            "She pushes the boundaries of what the project can do."
        ),
        "personality": "Brilliant, impatient, technically demanding.",
    },
    "Oscar": {
        "role": Role.CONTRIBUTOR,
        "expertise": ExpertiseLevel.INTERMEDIATE,
        "backstory": (
            "Oscar is a web developer who contributes frontend bug fixes and "
            "accessibility improvements. He cares deeply about making the "
            "project usable for everyone."
        ),
        "personality": "Empathetic, accessibility-focused, steady.",
    },
    "Hana": {
        "role": Role.KNOWLEDGE_CURATOR,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Hana is a former teacher who brings pedagogical skill to "
            "documentation. She creates step-by-step tutorials that have "
            "significantly improved the project's onboarding experience."
        ),
        "personality": "Patient, structured, excellent communicator.",
    },
    "Viktor": {
        "role": Role.MAINTAINER,
        "expertise": ExpertiseLevel.EXPERT,
        "backstory": (
            "Viktor is a veteran open-source contributor who has maintained "
            "several large projects. He brings governance experience and often "
            "mediates disagreements between contributors."
        ),
        "personality": "Diplomatic, experienced, values consensus.",
    },
    "Preet": {
        "role": Role.CONTRIBUTOR,
        "expertise": ExpertiseLevel.SENIOR,
        "backstory": (
            "Preet is a full-stack developer who contributes bug fixes across "
            "the entire stack. She is versatile and willing to take on any "
            "task that needs doing, even if it's not glamorous."
        ),
        "personality": "Versatile, selfless, quietly effective.",
    },
}


def get_all_agent_profiles(community_size: int | None = None) -> dict:
  """Return agent profiles, including extras if community_size > 16."""
  profiles = dict(AGENT_PROFILES)
  if community_size is not None and community_size > len(profiles):
    profiles.update(EXTRA_AGENT_PROFILES)
  return profiles
