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

"""VLLM Remote Language Model adapter.

Connects to a vLLM server running the OpenAI-compatible API over HTTP.
Use this when vLLM runs as a separate service (e.g. on DGX Spark).
For in-process vLLM, see concordia.contrib.language_models.vllm.
"""

from collections.abc import Collection, Sequence
import requests
import json
from typing import Any, Mapping
from concordia.language_model import language_model
from concordia.utils import sampling

class VLLMModel(language_model.LanguageModel):
  """Adapter for vLLM's OpenAI-compatible API."""

  def __init__(
      self,
      model_name: str,
      api_base: str = "http://localhost:8000/v1",
  ):
    self._model_name = model_name
    self._api_url = f"{api_base}/completions"

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    payload = {
        "model": self._model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": list(terminators) if terminators else None,
        "seed": seed,
    }
    
    response = requests.post(self._api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["text"]

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    # Use standard sampling helper to extract a valid choice from the model's text response
    return sampling.sample_choice(
        self,
        prompt,
        responses,
        seed=seed,
    )
