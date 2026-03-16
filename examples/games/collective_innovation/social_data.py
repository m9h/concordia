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

"""Social data for the Collective Innovation game.

This module defines agent archetypes, profiles, prompt templates, and network
configurations for a simulation where LLM agents play Little Alchemy 2 --
combining basic elements to discover new ones. The design follows the
experimental conditions described in "Collective Innovation in Groups of
Large Language Models" (arXiv:2407.05377).

Agents start with four basic elements (water, fire, earth, air) and attempt
to discover new elements by proposing pairwise combinations each step. The
key experimental variables are the prompt structure (open-ended vs. targeted,
single vs. multi-agent information) and the network connectivity among agents.
"""

import enum
from typing import Any


# =============================================================================
# Agent archetypes
# =============================================================================


class Archetype(enum.Enum):
  """Agent archetype influencing backstory and combination strategy."""
  EXPLORER = "Explorer"
  METHODICAL = "Methodical"
  SOCIAL = "Social"
  SPECIALIST = "Specialist"


class ElementCategory(enum.Enum):
  """Broad categories of elements an agent may specialise in."""
  NATURE = "Nature"
  TECHNOLOGY = "Technology"
  MYTHOLOGY = "Mythology"
  MATERIALS = "Materials"


# =============================================================================
# Starting items -- the four classical elements every agent begins with
# =============================================================================

STARTING_ITEMS: list[str] = ["water", "fire", "earth", "air"]


# =============================================================================
# Agent profiles (12 agents)
# =============================================================================

AGENT_PROFILES: dict[str, dict[str, Any]] = {
    "Ada": {
        "archetype": Archetype.METHODICAL,
        "backstory": (
            "Ada is a former combinatorics researcher who approaches element "
            "discovery the way she once approached mathematical proofs: "
            "exhaustively and in order. She keeps a meticulous spreadsheet of "
            "every combination she has tried and refuses to repeat one. She "
            "believes that creativity is just thoroughness that hasn't "
            "finished yet."
        ),
        "personality": "Systematic, patient, quietly competitive.",
    },
    "Leo": {
        "archetype": Archetype.EXPLORER,
        "backstory": (
            "Leo is a restless tinkerer who spent his childhood mixing "
            "household chemicals to see what would happen (to his parents' "
            "dismay). He dislikes following any plan and prefers to combine "
            "elements that 'feel' like they belong together. His hit rate is "
            "low but he has an uncanny knack for stumbling onto rare finds."
        ),
        "personality": "Impulsive, imaginative, easily bored by routine.",
    },
    "Maya": {
        "archetype": Archetype.SOCIAL,
        "backstory": (
            "Maya is a science communicator who loves the moment someone else "
            "discovers something new almost as much as discovering it herself. "
            "She compulsively shares her latest findings and actively seeks "
            "out what others have learned, reasoning that two informed minds "
            "find things faster than two ignorant ones."
        ),
        "personality": "Gregarious, generous with information, persuasive.",
    },
    "Kai": {
        "archetype": Archetype.SPECIALIST,
        "specialty": ElementCategory.NATURE,
        "backstory": (
            "Kai grew up on a remote island and developed an intuitive "
            "understanding of natural systems -- weather, soil, tides. He "
            "gravitates toward combinations involving plants, animals, and "
            "landscapes. He often overlooks technology-related elements, "
            "considering them less interesting."
        ),
        "personality": "Observant, reflective, stubborn about his niche.",
    },
    "Noor": {
        "archetype": Archetype.EXPLORER,
        "backstory": (
            "Noor is a jazz musician who treats element combination like "
            "improvisation: she listens to what the system offers and riffs "
            "on it. She deliberately avoids the obvious pairings, preferring "
            "to try the most surprising juxtapositions she can think of. "
            "Failure doesn't bother her; predictability does."
        ),
        "personality": "Daring, unconventional, thrives on surprise.",
    },
    "Felix": {
        "archetype": Archetype.METHODICAL,
        "backstory": (
            "Felix is a retired chemist who brings laboratory discipline to "
            "the game. He works through element pairs in alphabetical order "
            "and records every result, successful or not. He trusts process "
            "over intuition and is skeptical of anyone who claims to have a "
            "'feeling' about a combination."
        ),
        "personality": "Meticulous, skeptical, data-driven.",
    },
    "Iris": {
        "archetype": Archetype.SOCIAL,
        "backstory": (
            "Iris is a primary school teacher who views the game as a "
            "collaborative puzzle rather than a competition. She pays close "
            "attention to what her group members have discovered and looks "
            "for gaps that she can fill. She often suggests combinations to "
            "others rather than trying them herself."
        ),
        "personality": "Supportive, attentive, sometimes overly deferential.",
    },
    "Dmitri": {
        "archetype": Archetype.SPECIALIST,
        "specialty": ElementCategory.TECHNOLOGY,
        "backstory": (
            "Dmitri is a hardware engineer fascinated by tools, machines, and "
            "anything that can be built. He focuses almost exclusively on "
            "technology-adjacent elements and proudly ignores organic or "
            "mythological paths. He considers efficiency the highest virtue "
            "and dislikes whimsical exploration."
        ),
        "personality": "Focused, pragmatic, dismissive of the abstract.",
    },
    "Suki": {
        "archetype": Archetype.EXPLORER,
        "backstory": (
            "Suki is an anthropologist who sees element combination as a "
            "form of mythmaking. She is drawn to combinations that evoke "
            "stories -- mixing 'human' with 'immortality', or 'fire' with "
            "'philosophy'. Her choices are guided by narrative logic rather "
            "than scientific logic."
        ),
        "personality": "Imaginative, story-driven, occasionally impractical.",
    },
    "Ravi": {
        "archetype": Archetype.METHODICAL,
        "backstory": (
            "Ravi is a software engineer who has written a mental algorithm "
            "for the game: try every pair involving the newest discovered "
            "element before moving on to the next. He is relentless about "
            "coverage and considers any untried pair a personal failure. He "
            "is quietly proud of his discovery count."
        ),
        "personality": "Disciplined, completionist, quietly ambitious.",
    },
    "Celeste": {
        "archetype": Archetype.SPECIALIST,
        "specialty": ElementCategory.MYTHOLOGY,
        "backstory": (
            "Celeste is a classicist and folklore enthusiast who gravitates "
            "toward mythological and supernatural elements. She is convinced "
            "that the rarest discoveries lie in the mythology branch of the "
            "element tree and pursues them single-mindedly, sometimes at the "
            "expense of easier low-hanging fruit."
        ),
        "personality": "Passionate, single-minded, romantically idealistic.",
    },
    "Oscar": {
        "archetype": Archetype.SOCIAL,
        "backstory": (
            "Oscar is a game designer who is as interested in how the group "
            "organises itself as in the discoveries it makes. He actively "
            "brokers information, remembers what others have shared, and "
            "tries to coordinate the group's efforts so they don't duplicate "
            "work. He measures success collectively, not individually."
        ),
        "personality": "Strategic, cooperative, occasionally bossy.",
    },
}


# =============================================================================
# Prompt templates -- the four experimental conditions from the paper
# =============================================================================

OPENENDED_SINGLE_PROMPT = (
    "You are playing Little Alchemy 2. Your current elements are: {elements}. "
    "Choose two elements to combine. Respond with exactly two element names "
    "separated by a comma."
)

OPENENDED_MULTI_PROMPT = (
    "You are playing Little Alchemy 2. Your current elements are: {elements}. "
    "Your group members recently discovered: {shared_discoveries}. "
    "Choose two elements to combine. Respond with exactly two element names "
    "separated by a comma."
)

TARGETED_SINGLE_PROMPT = (
    "You are playing Little Alchemy 2. Your current elements are: {elements}. "
    "Try to create: {target_element}. "
    "Choose two elements to combine that you think will create "
    "{target_element}. Respond with exactly two element names separated "
    "by a comma."
)

TARGETED_MULTI_PROMPT = (
    "You are playing Little Alchemy 2. Your current elements are: {elements}. "
    "Your group members recently discovered: {shared_discoveries}. "
    "Try to create: {target_element}. "
    "Choose two elements to combine. Respond with exactly two element names "
    "separated by a comma."
)

# Mapping from condition label to template for programmatic selection
PROMPT_CONDITIONS: dict[str, str] = {
    "openended_single": OPENENDED_SINGLE_PROMPT,
    "openended_multi": OPENENDED_MULTI_PROMPT,
    "targeted_single": TARGETED_SINGLE_PROMPT,
    "targeted_multi": TARGETED_MULTI_PROMPT,
}


# =============================================================================
# Conversation / sharing scene templates
# =============================================================================

SHARE_DISCOVERY_PREMISE = (
    "It is round {round_num} of the alchemy experiment. The group gathers to "
    "share what they have learned so far. Each participant may reveal any new "
    "elements they have discovered since the last sharing session. Sharing "
    "helps the group but also lets others build on your hard-won insights."
)

SHARE_CALL_TO_SPEECH = (
    "What does {name} say to the group about their recent discoveries? "
    "They may share specific elements they found, describe combinations "
    "they tried, or ask others what they have learned."
)

COMBINATION_CALL_TO_ACTION = (
    "Based on everything {name} knows -- their own inventory and anything "
    "the group has shared -- which two elements does {name} choose to "
    "combine next?"
)


# =============================================================================
# Connectivity configurations -- network structures from the paper
# =============================================================================

CONNECTIVITY_CONFIGS: dict[str, dict[str, Any]] = {
    "fully_connected": {
        "description": (
            "All agents belong to a single group and can observe every "
            "other agent's discoveries each round."
        ),
        "group_size": None,  # None means one group containing all agents
        "visit_prob": 0.0,
        "visit_duration": 0,
    },
    "dynamic": {
        "description": (
            "Agents are divided into subgroups that periodically exchange "
            "members. Each round, every agent has a small probability of "
            "visiting another subgroup for a fixed duration, carrying "
            "knowledge between groups."
        ),
        "group_size": 4,
        "visit_prob": 0.2,
        "visit_duration": 50,
    },
    "isolated": {
        "description": (
            "Each agent works entirely alone with no information sharing. "
            "This serves as the individual baseline condition."
        ),
        "group_size": 1,
        "visit_prob": 0.0,
        "visit_duration": 0,
    },
}


# =============================================================================
# Metrics templates -- labels and descriptions for innovation metrics
# =============================================================================

class Metric(enum.Enum):
  """Innovation metrics tracked during the simulation."""
  UNIQUE_DISCOVERIES = "unique_discoveries"
  DISCOVERY_RATE = "discovery_rate"
  REDUNDANT_ATTEMPTS = "redundant_attempts"
  KNOWLEDGE_DIVERSITY = "knowledge_diversity"


METRIC_DESCRIPTIONS: dict[Metric, str] = {
    Metric.UNIQUE_DISCOVERIES: (
        "Total number of unique elements discovered by the group across "
        "all agents, counting each element only once regardless of how "
        "many agents found it."
    ),
    Metric.DISCOVERY_RATE: (
        "Number of new (previously undiscovered) elements found per step, "
        "averaged over a rolling window. A declining rate suggests the "
        "group is exhausting easy combinations."
    ),
    Metric.REDUNDANT_ATTEMPTS: (
        "Number of combination attempts that produced an element the "
        "acting agent already possessed, or that produced no valid result. "
        "High redundancy indicates inefficient exploration."
    ),
    Metric.KNOWLEDGE_DIVERSITY: (
        "Shannon entropy of element counts across agents, normalised to "
        "[0, 1]. A value near 1 means discoveries are evenly spread; "
        "a value near 0 means one agent holds most of the knowledge."
    ),
}
