# LLAMOSC Baseline

**LLM Agent-Based Modeling for Open Source Communities** - baseline implementation from the [OREL-group/GSoC](https://github.com/OREL-group/GSoC/tree/main/Open%20Source%20Sustainibility%20using%20LLMs) project (Vidhi Rohira, GSoC 2025).

This is a uv-managed, easy-to-run package extracted from the original LLAMOSC codebase for use as a comparison baseline against the Concordia SustainHub variant.

## Quick Start

```bash
# Test mode (no LLM needed, uses mock data):
uv run llamosc-headless --test

# Reproducible test run:
uv run llamosc-headless --test --seed 42

# Full LLM run (requires Ollama with llama3):
uv run llamosc-headless --contributors 5 --maintainers 3 --issues 5

# Decentralized algorithm:
uv run llamosc-headless --algorithm d --test

# Collaborative algorithm:
uv run llamosc-headless --algorithm c --test
```

## Algorithms

- **a** (Authoritarian): Maintainer rates contributors and assigns tasks (benevolent dictator model)
- **d** (Decentralized): Contributors bid on tasks, highest bid wins (meritocratic model)
- **c** (Collaborative): Team formation with Lead/Reviewer/Support roles

## Output

Results are saved to `output/`:
- `llamosc_results.json` - Full simulation data and metrics
- `llamosc_metrics.png` - Experience, code quality, and motivation plots

## Reference

- Blog: https://vidhirohira.github.io/blog/2025/08/25/final-evaluation.html
- Code: https://github.com/OREL-group/GSoC/tree/main/Open%20Source%20Sustainibility%20using%20LLMs
