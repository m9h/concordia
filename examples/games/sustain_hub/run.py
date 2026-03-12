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
flags.DEFINE_bool('skip_backstory', False, 'Whether to skip initial backstory generation (faster).')
flags.DEFINE_bool('verbose', False, 'Whether to print detailed simulation logs.')
flags.DEFINE_string('project', None, 'GCP Project ID for Vertex AI.')
flags.DEFINE_string('location', 'us-central1', 'GCP Location for Vertex AI.')
flags.DEFINE_bool('use_active_inference', True, 'Whether to use Active Inference agents.')
flags.DEFINE_bool('fast', False, 'Skip conversation scenes, go straight to task decisions.')
flags.DEFINE_string('vllm_url', None, 'vLLM API base URL (e.g. http://localhost:8000/v1).')


def main(argv):
  del argv  # Unused.

  from concordia.contrib.language_models.google import gemini_model
  from concordia.language_model import retry_wrapper
  from concordia.testing import mock_model

  if FLAGS.use_mock:
    model = mock_model.MockModel()
  elif FLAGS.vllm_url:
    from concordia.contrib.language_models import vllm_remote
    model = vllm_remote.VLLMModel(
        model_name=FLAGS.model_name,
        api_base=FLAGS.vllm_url,
    )
    model = retry_wrapper.RetryLanguageModel(
        model,
        retry_tries=5,
        retry_delay=2.0,
        backoff_factor=1.5,
    )
  else:
    api_key = FLAGS.api_key or os.environ.get('GEMINI_API_KEY', '')
    if not api_key and not FLAGS.project:
      print('Error: GEMINI_API_KEY not found. Use --use_mock for testing, '
            '--project for Vertex, or --vllm_url for local vLLM.')
      return

    model = gemini_model.GeminiModel(
        model_name=FLAGS.model_name,
        api_key=api_key if not FLAGS.project else None,
        project=FLAGS.project,
        location=FLAGS.location,
    )
    model = retry_wrapper.RetryLanguageModel(
        model,
        retry_tries=10,
        retry_delay=5.0,
        backoff_factor=2.0,
    )

  try:
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embedder = lambda x: st_model.encode(x, show_progress_bar=False)
  except ImportError:
    print(
        'sentence-transformers not installed. '
        'Run: pip install sentence-transformers'
    )
    return

  print('=' * 72)
  print('SustainHub: Open-Source Community Sustainability Simulation')
  print('=' * 72)
  print(f'Sprints: {FLAGS.num_sprints}')
  print(f'Community Size: {FLAGS.community_size}')
  print(f'Stress scenarios: {FLAGS.enable_stress}')
  print(f'Active Inference: {FLAGS.use_active_inference}')
  if FLAGS.vllm_url:
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
      enable_stress=FLAGS.enable_stress,
      community_size=FLAGS.community_size,
      skip_backstory=FLAGS.skip_backstory,
      verbose=FLAGS.verbose,
      use_active_inference=FLAGS.use_active_inference,
      skip_conversation=FLAGS.fast,
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
