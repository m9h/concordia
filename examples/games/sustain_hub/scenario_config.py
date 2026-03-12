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

"""Scenario configuration for SustainHub simulation.

This module defines the parameters for running the SustainHub open-source
community sustainability simulation. Modify these values to create different
experimental conditions.
"""

# =============================================================================
# Simulation parameters
# =============================================================================

# Number of sprints to simulate
NUM_SPRINTS = 3

# Random seed (set to None for non-deterministic runs)
SEED = 42

# Whether to include stress scenarios (contributor dropout, task overload)
ENABLE_STRESS = True

# =============================================================================
# Agent selection
# =============================================================================

# Available agents (from social_data.AGENT_PROFILES):
#   Priya     - Contributor (Senior)
#   Marcus    - Contributor (Intermediate)
#   Anya      - Innovator (Expert)
#   Jordan    - Innovator (Intermediate)
#   Elena     - Knowledge Curator (Senior)
#   Raj       - Maintainer (Expert)
#   Lin       - Maintainer (Senior)
#   Sam       - Contributor (Apprentice)

# Use all 8 agents (set to a subset for faster runs)
AGENTS_TO_USE = None  # None = all agents

# For a minimal 4-agent run (one of each role):
# AGENTS_TO_USE = ["Priya", "Anya", "Elena", "Raj"]

# =============================================================================
# Experimental conditions
# =============================================================================

# Condition 1: Baseline (no stress)
# ENABLE_STRESS = False
# NUM_SPRINTS = 3

# Condition 2: With stress (default)
# ENABLE_STRESS = True
# NUM_SPRINTS = 3

# Condition 3: Extended run (observe long-term learning)
# ENABLE_STRESS = True
# NUM_SPRINTS = 5

# Condition 4: Minimal team under pressure
# AGENTS_TO_USE = ["Priya", "Anya", "Elena", "Raj"]
# ENABLE_STRESS = True
# NUM_SPRINTS = 4
