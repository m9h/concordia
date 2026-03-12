# Technical Scaling Considerations

## Concordia Computational Cost Model

### LLM Calls Per Step

| Engine | Fixed Overhead | Per-Entity Cost | Total (N entities) |
|--------|---------------|-----------------|-------------------|
| Sequential | 5 calls (terminate, next_gm, next_acting, action_spec, resolve) | 2 calls (observe + act) | 5 + 2N |
| Simultaneous | 3 calls (terminate, next_gm, next_acting) | 4 calls (action_spec + observe + act + resolve) | 3 + 4N (parallelized) |
| Asynchronous | 0 (no sync) | Independent loops | N concurrent streams |

### Cost at Scale

| Community Size | Engine | LLM Calls/Step | Steps/Sprint | Sprints | Total LLM Calls |
|---------------|--------|----------------|-------------|---------|-----------------|
| 4 agents | Sequential | 13 | ~10 | 1 | ~130 |
| 8 agents | Sequential | 21 | ~15 | 3 | ~945 |
| 16 agents | Sequential | 37 | ~20 | 5 | ~3,700 |
| 16 agents | Simultaneous | 67 | ~20 | 5 | ~6,700 (but faster) |

**Note:** Each agent has ~5 context components, each potentially doing 1 memory
retrieval (embedding call) per pre_act phase. Add ~5N embedding calls per step.

---

## Rate Limiting Bottlenecks

### Current Mitigations
1. **Sequential engine hardcoded sleep:** 1 second per step (sequential.py:344)
2. **Gemini periodic sleep:** 10s pause every 10 API calls (gemini_model.py)
3. **Retry wrapper:** 10 retries, 5s initial delay, 2x exponential backoff, 300s max

### API Rate Limits by Provider

| Provider | Model | RPM | TPM | Cost/M tokens | Latency |
|----------|-------|-----|-----|---------------|---------|
| Google AI Studio (free) | Gemini 2.0 Flash | 15 | 1M | $0 | ~1-3s |
| Google AI Studio (paid) | Gemini 2.0 Flash | 2000 | 4M | $0.075 in / $0.30 out | ~0.5-1s |
| Google AI Studio (paid) | Gemini 2.0 Flash Lite | 4000 | 4M | $0.075 in / $0.30 out | ~0.3-0.5s |
| Together AI | Qwen 2.5-7B | 600 | - | $0.20 | ~0.5s |
| Groq | Llama 3.1-8B | 30 | 6K | $0.05 in / $0.08 out | ~0.1s |

### Throughput Estimates

**Gemini 2.0 Flash (paid tier), 16 agents, sequential engine:**
- 37 LLM calls per step, ~1s latency each
- With 1s sleep: ~38s per step
- 20 steps/sprint x 5 sprints = 100 steps
- **Total: ~63 minutes per simulation run**
- Cost: ~3,700 calls x ~500 tokens avg = ~1.85M tokens = ~$0.14

**Local vLLM (Qwen 2.5-7B on DGX Spark), 16 agents:**
- 37 calls/step, ~0.05s latency (batch), no rate limit
- ~2s per step (dominated by generation time)
- 100 steps = **~200 seconds (3.3 minutes)**
- Cost: $0 (hardware already available)
- **18x speedup over cloud API**

---

## Scaling Strategies

### Tier 1: Cloud API (current)
- **Best for:** Prototyping, hackathon demos, <8 agents
- **Bottleneck:** Rate limits (15 RPM free tier)
- **Cost:** $0.03-0.15 per run
- **Mitigation:** Use Flash Lite, batch with sweep_runner.py

### Tier 2: Local vLLM on DGX Spark
- **Best for:** Parameter sweeps, 8-16 agents, autoresearch loop
- **Bottleneck:** Model quality (7B vs 70B+)
- **Cost:** $0 marginal (hardware sunk cost)
- **Setup:** `vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1`
- **Throughput:** 100-370 tok/s, no rate limits
- **Connection:** OpenAI-compatible endpoint → Concordia's OpenAI model wrapper

### Tier 3: Hybrid (Dual-Process)
- **Best for:** Large-scale research, 16+ agents, long runs
- **System 1:** JaxMARL on DGX Spark GPU — 10^4 FPS for parameter sweeps
- **System 2:** Concordia + LLM — triggered by phase transitions
- **Bottleneck:** StateTranslator middleware (not yet built)
- **Cost:** Minimal — GPU handles bulk, LLM only for critical moments

### Tier 4: Multi-GPU Distributed
- **Best for:** Population-scale experiments (100+ agents)
- **Not yet needed:** Current research fits in Tier 2-3

---

## Memory and Embedding Scaling

### Current Implementation
- **Embedder:** sentence-transformers/all-mpnet-base-v2 (local, ~768-dim)
- **Storage:** In-memory pandas DataFrame with cosine similarity retrieval
- **Cost:** 1 embedding call per memory add + 1 per retrieval
- **Scaling:** O(N) retrieval over all stored memories (no index)

### Bottleneck Analysis
For 16 agents x 20 steps x 5 sprints x 5 components:
- ~8,000 embedding calls (add + retrieve)
- At ~1ms each (local model): **~8 seconds total** — NOT a bottleneck
- Memory DataFrame grows to ~1,600 rows — trivial for cosine similarity

### Future: Vector DB
Only needed at 100+ agents or 50+ sprints. Options:
- FAISS (GPU-accelerated, ~1M vectors in milliseconds)
- ChromaDB (simple API, good for prototyping)

---

## Key Technical Gaps

1. **No batching in Concordia's embedding layer** — each add/retrieve is
   independent. For cloud embeddings this would be costly; local is fine.

2. **Sequential engine's 1s sleep is hardcoded** — should be configurable
   or removed when using local models.

3. **No vLLM/SGLang backend exists yet** — must use OpenAI-compatible wrapper.
   Need to verify compatibility with Concordia's OpenAI model class.

4. **Component parallelization limited by Python GIL** — I/O-bound LLM calls
   parallelize well, but memory-bound operations don't. Async engine helps.

5. **No checkpointing mid-simulation** — if a run crashes at sprint 4 of 5,
   must restart. Need to add sprint-level checkpointing.

---

## Recommendations for FTC Hackathon (March 14-15)

1. **Use Gemini Flash Lite (paid tier)** — 4000 RPM, cheapest, fastest cloud option
2. **8 agents, 3 sprints** — completes in ~5 minutes, hits the social dilemma sweet spot
3. **experiments.py for live demos** — runs in <1 second, no API needed
4. **Pre-compute one full Concordia run** — save HTML log + results.json for display
5. **Streamlit dashboard** — show experiment ladder results + one Concordia narrative side by side
