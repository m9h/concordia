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

"""Runner script for the SustainHub simulation."""

import os
import sys
from absl import app
from absl import flags
from examples.games.sustain_hub import simulation

FLAGS = flags.FLAGS

flags.DEFINE_string('model_name', 'gemini-2.0-flash', 'Name of the LLM to use.')
flags.DEFINE_string('api_key', None, 'API key (or set GEMINI_API_KEY env var).')
flags.DEFINE_string(
    'output_dir', '/tmp/sustain_hub_results', 'Directory for output files.'
)
flags.DEFINE_bool('use_mock', False, 'Use a mock model for testing.')
flags.DEFINE_integer('num_sprints', 3, 'Number of sprints to run.')
flags.DEFINE_bool('enable_stress', True, 'Whether to enable stress scenarios.')
flags.DEFINE_integer('community_size', 16, 'Number of agents in the community.')
flags.DEFINE_integer('seed', None, 'Random seed.')
flags.DEFINE_bool('skip_backstory', False, 'Whether to skip initial backstory generation (faster).')
flags.DEFINE_bool('verbose', False, 'Whether to print detailed simulation logs.')
flags.DEFINE_string('project', None, 'GCP Project ID for Vertex AI.')
flags.DEFINE_string('location', 'us-central1', 'GCP Location for Vertex AI.')
flags.DEFINE_bool('use_active_inference', True, 'Whether to use Active Inference agents.')
flags.DEFINE_bool('fast', False, 'Skip conversation scenes, go straight to task decisions.')
flags.DEFINE_string('governance', 'free_choice',
                    'Governance mode: free_choice, dictator, meritocratic.')
flags.DEFINE_string('vllm_url', None, 'vLLM API base URL (e.g. http://localhost:8000/v1).')
flags.DEFINE_bool('nvidia_nim', False, 'Use NVIDIA NIM API (set NVIDIA_API_KEY env var).')
flags.DEFINE_bool('together', False, 'Use Together AI API (set TOGETHER_AI_API_KEY env var).')
flags.DEFINE_bool('groq', False, 'Use Groq API (set GROQ_API_KEY env var).')


def main(argv):
  del argv  # Unused.

  from concordia.language_model import retry_wrapper
  from concordia.testing import mock_model

  if FLAGS.use_mock:
    model = mock_model.MockModel()
  elif FLAGS.groq:
    from concordia.contrib.language_models.groq import groq_model
    groq_key = os.environ.get('GROQ_API_KEY', '')
    if not groq_key:
      print('Error: GROQ_API_KEY not set.', file=sys.stderr)
      sys.exit(1)
    model = groq_model.GroqModel(
        model_name=FLAGS.model_name,
        api_key=groq_key,
    )
    model = retry_wrapper.RetryLanguageModel(
        model, retry_tries=5, retry_delay=2.0, backoff_factor=2.0,
    )
  elif FLAGS.nvidia_nim:
    from concordia.contrib.language_models import vllm_remote
    nim_key = os.environ.get('NVIDIA_API_KEY', '')
    if not nim_key:
      print('Error: NVIDIA_API_KEY not set. Get a free key at build.nvidia.com',
            file=sys.stderr)
      sys.exit(1)
    nim_model = FLAGS.model_name if '/' in FLAGS.model_name else 'meta/llama-3.1-8b-instruct'
    model = vllm_remote.VLLMModel(
        model_name=nim_model,
        api_base='https://integrate.api.nvidia.com/v1',
        api_key=nim_key,
        chat_mode=True,
    )
    model = retry_wrapper.RetryLanguageModel(
        model, retry_tries=8, retry_delay=3.0, backoff_factor=2.0,
    )
  elif FLAGS.together:
    from concordia.contrib.language_models.together import together_ai_model
    model = together_ai_model.Base(model_name=FLAGS.model_name)
    model = retry_wrapper.RetryLanguageModel(
        model, retry_tries=8, retry_delay=3.0, backoff_factor=2.0,
    )
  elif FLAGS.vllm_url:
    from concordia.contrib.language_models import vllm_remote
    # Auto-detect API key and chat mode for known hosted endpoints
    vllm_api_key = FLAGS.api_key
    vllm_chat_mode = False
    if 'nvidia.com' in FLAGS.vllm_url:
      vllm_api_key = vllm_api_key or os.environ.get('NVIDIA_API_KEY', '')
      vllm_chat_mode = True
      if not vllm_api_key:
        print('Error: NVIDIA_API_KEY not set for NIM endpoint. '
              'Get a free key at build.nvidia.com', file=sys.stderr)
        sys.exit(1)
    model = vllm_remote.VLLMModel(
        model_name=FLAGS.model_name,
        api_base=FLAGS.vllm_url,
        api_key=vllm_api_key or None,
        chat_mode=vllm_chat_mode,
    )
    model = retry_wrapper.RetryLanguageModel(
        model, retry_tries=8, retry_delay=3.0, backoff_factor=2.0,
    )
  else:
    api_key = FLAGS.api_key or os.environ.get('GEMINI_API_KEY', '')
    if not api_key and not FLAGS.project:
      print('Error: Set GEMINI_API_KEY, or use --nvidia_nim, --together, '
            '--vllm_url, or --use_mock.', file=sys.stderr)
      sys.exit(1)

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

  print('=' * 72)
  print('SustainHub: Open-Source Community Sustainability Simulation')
  print('=' * 72)
  print(f'Sprints: {FLAGS.num_sprints}')
  print(f'Community Size: {FLAGS.community_size}')
  print(f'Stress scenarios: {FLAGS.enable_stress}')
  print(f'Active Inference: {FLAGS.use_active_inference}')
  print(f'Governance: {FLAGS.governance}')
  if FLAGS.nvidia_nim:
    print(f'Backend: NVIDIA NIM ({FLAGS.model_name})')
  elif FLAGS.together:
    print(f'Backend: Together AI ({FLAGS.model_name})')
  elif FLAGS.vllm_url:
    print(f'Backend: vLLM ({FLAGS.vllm_url})')
  elif FLAGS.project:
    print(f'Backend: Vertex AI (Project: {FLAGS.project})')
  else:
    print('Backend: AI Studio')
  print('=' * 72)

  results = simulation.run_simulation(
      model=model,
      embedder=embedder,
      num_sprints=FLAGS.num_sprints,
      seed=FLAGS.seed,
      enable_stress=FLAGS.enable_stress,
      community_size=FLAGS.community_size,
      skip_backstory=FLAGS.skip_backstory,
      verbose=FLAGS.verbose,
      use_active_inference=FLAGS.use_active_inference,
      skip_conversation=FLAGS.fast,
      governance=FLAGS.governance,
  )

  # Print summary
  print('\n' + '=' * 72)
  print('SIMULATION RESULTS')
  print('=' * 72)

  print(f'\nFinal Harmony Index: {results["harmony_index"]:.3f}')
  print(f'Resilience Quotient: {results["resilience_quotient"]:.3f}')

  print('\nCumulative Scores by Agent:')
  for name, score in sorted(
      results['scores'].items(), key=lambda x: x[1], reverse=True
  ):
    role = results['player_roles'].get(name, 'Unknown')
    print(f'  {name:12s} ({role:12s}): {score:+.1f}')

  print('\nSprint-by-Sprint Harmony Index:')
  for i, sprint in enumerate(results['sprint_history']):
    print(f'  Sprint {i + 1}: HI = {sprint["harmony_index"]:.3f}')
    for name, task in sprint['joint_action'].items():
      score = sprint['scores'].get(name, 0.0)
      task_str = task[:50] if task else "None"
      print(f'    {name:12s} -> {task_str:50s} ({score:+.1f})')

  # Save results
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  output_file = os.path.join(FLAGS.output_dir, 'results.json')
  # Filter out non-serializable objects
  serializable_results = {
      k: v for k, v in results.items() if k != 'structured_log'
  }
  import json
  with open(output_file, 'w') as f:
    json.dump(serializable_results, f, indent=2)
  print(f'\nResults saved to {output_file}')

  # Also save to a fixed location for the live dashboard
  with open('live_results.json', 'w') as f:
    json.dump(serializable_results, f, indent=2)

  html_file = os.path.join(FLAGS.output_dir, 'simulation_log.html')
  html_content = results['structured_log'].to_html()
  with open(html_file, 'w') as f:
    f.write(html_content)
  print(f'HTML log saved to {html_file}')


if __name__ == '__main__':
  app.run(main)
