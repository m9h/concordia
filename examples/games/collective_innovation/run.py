#!/usr/bin/env python3
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

"""Runner script for the Collective Innovation simulation."""

import json
import os

from absl import app
from absl import flags

from examples.games.collective_innovation import simulation

FLAGS = flags.FLAGS

# Model/backend flags
flags.DEFINE_string('model_name', 'gemini-2.0-flash', 'Name of the LLM to use.')
flags.DEFINE_string('api_key', None, 'API key (or set GEMINI_API_KEY env var).')
flags.DEFINE_string(
    'output_dir', '/tmp/collective_innovation_results',
    'Directory for output files.',
)
flags.DEFINE_bool('use_mock', False, 'Use a mock model for testing.')
flags.DEFINE_string(
    'vllm_url', None,
    'vLLM API base URL (e.g. http://localhost:8000/v1).',
)
flags.DEFINE_bool(
    'nvidia_nim', False,
    'Use NVIDIA NIM API (set NVIDIA_API_KEY env var).',
)
flags.DEFINE_bool(
    'together', False,
    'Use Together AI API (set TOGETHER_AI_API_KEY env var).',
)
flags.DEFINE_bool('groq', False, 'Use Groq API (set GROQ_API_KEY env var).')
flags.DEFINE_string('project', None, 'GCP Project ID for Vertex AI.')
flags.DEFINE_string('location', 'us-central1', 'GCP Location for Vertex AI.')

# Simulation-specific flags
flags.DEFINE_integer('num_steps', 50, 'Number of combination steps to simulate.')
flags.DEFINE_integer('num_agents', 6, 'Number of agents.')
flags.DEFINE_string(
    'connectivity', 'fully_connected',
    'Network structure: fully_connected, dynamic, isolated',
)
flags.DEFINE_string(
    'prompt_mode', 'openended_multi',
    'Prompt variant: openended_single, openended_multi, '
    'targeted_single, targeted_multi',
)
flags.DEFINE_integer('seed', None, 'Random seed.')
flags.DEFINE_bool(
    'skip_backstory', False,
    'Skip initial backstory generation.',
)
flags.DEFINE_bool('fast', False, 'Skip conversation scenes.')
flags.DEFINE_bool('verbose', False, 'Print detailed logs.')


def _get_backend_label() -> str:
  """Return a human-readable label for the active backend."""
  if FLAGS.use_mock:
    return 'Mock'
  if FLAGS.groq:
    return f'Groq ({FLAGS.model_name})'
  if FLAGS.nvidia_nim:
    return f'NVIDIA NIM ({FLAGS.model_name})'
  if FLAGS.together:
    return f'Together AI ({FLAGS.model_name})'
  if FLAGS.vllm_url:
    return f'vLLM ({FLAGS.vllm_url})'
  if FLAGS.project:
    return f'Vertex AI (Project: {FLAGS.project})'
  return f'AI Studio ({FLAGS.model_name})'


def main(argv):
  del argv  # Unused.

  # --------------- model setup ---------------
  from concordia.language_model import retry_wrapper
  from concordia.testing import mock_model

  if FLAGS.use_mock:
    model = mock_model.MockModel()
  elif FLAGS.groq:
    groq_key = os.environ.get('GROQ_API_KEY', '')
    if not groq_key:
      print('Error: GROQ_API_KEY not set.')
      return
    from concordia.contrib.language_models.groq import groq_model
    model = groq_model.GroqModel(
        model_name=FLAGS.model_name,
        api_key=groq_key,
    )
    model = retry_wrapper.RetryLanguageModel(
        model, retry_tries=5, retry_delay=2.0, backoff_factor=2.0,
    )
  elif FLAGS.nvidia_nim:
    from concordia.contrib.language_models import vllm_remote
    nim_key = (os.environ.get('NVIDIA_API_KEY', '')
               or os.environ.get('NGC_API_KEY', ''))
    if not nim_key:
      print('Error: NVIDIA_API_KEY or NGC_API_KEY not set. '
            'Get a free key at build.nvidia.com')
      return
    nim_model = (
        FLAGS.model_name
        if '/' in FLAGS.model_name
        else 'meta/llama-3.1-8b-instruct'
    )
    model = vllm_remote.VLLMModel(
        model_name=nim_model,
        api_base='https://integrate.api.nvidia.com/v1',
        api_key=nim_key,
        chat_mode=True,
    )
    model = retry_wrapper.RetryLanguageModel(
        model, retry_tries=15, retry_delay=5.0, backoff_factor=2.0,
    )
  elif FLAGS.together:
    from concordia.contrib.language_models.together import together_ai_model
    model = together_ai_model.Base(model_name=FLAGS.model_name)
    model = retry_wrapper.RetryLanguageModel(
        model, retry_tries=8, retry_delay=3.0, backoff_factor=2.0,
    )
  elif FLAGS.vllm_url:
    from concordia.contrib.language_models import vllm_remote
    model = vllm_remote.VLLMModel(
        model_name=FLAGS.model_name,
        api_base=FLAGS.vllm_url,
    )
    model = retry_wrapper.RetryLanguageModel(
        model, retry_tries=8, retry_delay=3.0, backoff_factor=2.0,
    )
  else:
    api_key = FLAGS.api_key or os.environ.get('GEMINI_API_KEY', '')
    if not api_key and not FLAGS.project:
      print('Error: Set GEMINI_API_KEY, or use --nvidia_nim, --together, '
            '--vllm_url, or --use_mock.')
      return
    from concordia.contrib.language_models.google import gemini_model
    model = gemini_model.GeminiModel(
        model_name=FLAGS.model_name,
        api_key=api_key if not FLAGS.project else None,
        project=FLAGS.project,
        location=FLAGS.location,
    )
    model = retry_wrapper.RetryLanguageModel(
        model, retry_tries=10, retry_delay=5.0, backoff_factor=2.0,
    )

  # --------------- embedder setup ---------------
  try:
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embedder = lambda x: st_model.encode(x, show_progress_bar=False)
  except Exception:
    # Fallback: lightweight sklearn-based embedder (works without torch)
    from sklearn.feature_extraction.text import HashingVectorizer
    print('Note: Using sklearn HashingVectorizer embedder (sentence-transformers unavailable).')
    _hv = HashingVectorizer(n_features=384, analyzer='char_wb', ngram_range=(2, 4), norm='l2')
    embedder = lambda x: _hv.transform([x]).toarray()[0]

  # --------------- banner ---------------
  backend = _get_backend_label()
  print('=' * 72)
  print('Collective Innovation: LLM Agents Playing Little Alchemy 2')
  print('=' * 72)
  print(f'Steps: {FLAGS.num_steps}')
  print(f'Agents: {FLAGS.num_agents}')
  print(f'Connectivity: {FLAGS.connectivity}')
  print(f'Prompt Mode: {FLAGS.prompt_mode}')
  print(f'Backend: {backend}')
  print('=' * 72)

  # --------------- run simulation ---------------
  results = simulation.run_simulation(
      model=model,
      embedder=embedder,
      num_steps=FLAGS.num_steps,
      num_agents=FLAGS.num_agents,
      connectivity=FLAGS.connectivity,
      prompt_mode=FLAGS.prompt_mode,
      seed=FLAGS.seed,
      skip_backstory=FLAGS.skip_backstory,
      skip_conversation=FLAGS.fast,
      verbose=FLAGS.verbose,
  )

  # --------------- results summary ---------------
  print('\n' + '=' * 72)
  print('SIMULATION RESULTS')
  print('=' * 72)

  print(f'\nTotal Unique Discoveries: {results["unique_discoveries"]}')
  print(f'Discovery Rate (per step): {results["discovery_rate"]:.3f}')
  print(f'Knowledge Diversity: {results["knowledge_diversity"]:.3f}')
  print(f'Innovation Score: {results["innovation_score"]:.3f}')

  print('\nPer-Agent Discovery Counts:')
  for name, info in sorted(
      results['per_agent_results'].items(),
      key=lambda x: x[1]['num_discoveries'],
      reverse=True,
  ):
    print(f'  {name:20s}: {info["num_discoveries"]} discoveries '
          f'({info["num_elements"]} elements)')

  discoveries = results.get('global_discoveries', [])
  if discoveries:
    print(f'\nLatest Discoveries (up to 10):')
    for elem in discoveries[-10:]:
      print(f'  {elem}')

  # --------------- save results ---------------
  os.makedirs(FLAGS.output_dir, exist_ok=True)

  serializable_results = {
      k: v for k, v in results.items() if k != 'structured_log'
  }
  output_file = os.path.join(FLAGS.output_dir, 'results.json')
  with open(output_file, 'w') as f:
    json.dump(serializable_results, f, indent=2)
  print(f'\nResults saved to {output_file}')

  html_file = os.path.join(FLAGS.output_dir, 'simulation_log.html')
  html_content = results['structured_log'].to_html()
  with open(html_file, 'w') as f:
    f.write(html_content)
  print(f'HTML log saved to {html_file}')


if __name__ == '__main__':
  app.run(main)
