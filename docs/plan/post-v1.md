# Post-V1 Roadmap

What comes after the MVP launch.

## Near-Term (v0.2 — Months 4-6)

### Online Learning for Transition Matrix

The v1 Markov matrix is static — trained offline, deployed frozen. v0.2 adds live updates:

- Track actual transitions in production conversations
- Update matrix weights with exponential decay (recent transitions weighted higher)
- Differential privacy: add calibrated noise to prevent leaking individual conversation patterns
- A/B test: static vs adaptive matrix, measure cache hit rate improvement

### Multi-Modal Retrieval

Skip STT for first-stage retrieval:

- Accept raw audio embeddings (wavRAG-style) directly
- Use a small audio encoder to produce a rough embedding from the audio stream
- Binary scan on audio embedding → candidate set
- Refine with text embedding at end-of-utterance
- Potential savings: eliminate STT latency (~200ms) from the retrieval path entirely

### Cross-Session Memory Integration

Layer on top of mem0/letta for user-specific prefetch:

- If the user has interacted before, load their typical topic patterns
- Personalized transition matrix: `P(next | current, user_history)`
- Example: returning customer always asks about billing → prefetch billing docs before they even start talking

### Additional Framework Plugins

- **Vapi** — plugin for Vapi's real-time voice platform
- **Retell AI** — integration with Retell's agent API
- **Bland AI** — integration for sales-focused deployments
- **OpenAI Realtime API** — adapter for OpenAI's voice API

## Medium-Term (v0.3 — Months 7-12)

### Hardware Acceleration

- **CUDA kernel** for binary signature scan at >10M scale
  - GPU excels at massively parallel hamming distance computation
  - Target: <0.1ms for 10M signatures on an A100
- **Apple Neural Engine (ANE)** path for Apple Silicon
  - Use CoreML for the embedder on M-series chips
  - ~2x speedup over CPU ONNX inference
- **Intel AMX** for matrix operations in the rescore step

### Scale to 10M+ Documents

- Hierarchical binary scan: cluster signatures into super-events → scan cluster representatives first
- Distributed event shards across NUMA nodes
- Tiered storage: hot events in RAM, cold events on NVMe with speculative prefetch

### Improved Predictor

Replace simple Markov with a lightweight neural predictor:

- ~2M parameter model (not an LLM)
- Takes: current event, last 5 events, conversation length, time-of-day
- Outputs: probability distribution over events
- Inference: ~1ms on CPU (vs ~10μs for Markov, vs ~200-500ms for LLM)
- Expected improvement: +10-15% hit rate on semi-structured domains

### Evaluation Framework

- Standardized eval suite for voice-AI retrieval
- Community-contributed domain benchmarks
- Leaderboard for retrieval latency + recall + cache hit rate

## Long-Term (v1.0 — Year 2+)

### Hosted Service (primd Cloud)

- Managed deployment with per-domain pretrained transition matrices
- SLA-backed p99 latency
- Usage-based pricing (per query, per warm_next call)
- Dashboard: cache hit rate, latency percentiles, topic distribution
- Auto-retraining: transition matrix updates from anonymized conversation patterns

### Edge Runtime

- Optimized build for IoT/embedded devices
- Target: Raspberry Pi 5, Jetson Orin Nano
- Use case: on-device voice assistants with local knowledge bases
- ~10K document corpus at <10ms retrieval

### Language-Agnostic Retrieval

- Multilingual embedding models (e.g., multilingual-MiniLM)
- Cross-language retrieval: query in Hindi, retrieve English docs
- Per-language binary quantization (separate PCA matrices per language)

### Ecosystem

- **primd-studio**: Web UI for corpus management, boundary tuning, matrix training
- **primd-eval**: Hosted benchmark service (upload corpus, get latency/recall numbers)
- **Marketplace**: Community-contributed transition matrices for specific domains
