# Market Research

## Voice AI Market Size

The voice AI market crossed $22B in 2026 and is growing at 20-39% CAGR across segments.

| Segment | 2024-2025 Value | Projected | CAGR |
|---|---|---|---|
| Voice AI Agents | $2.4B | $47.5B by 2034 | 34.8% |
| AI Voice Generator | $4.16B | $20.71B by 2031 | 30.7% |
| Voice & Language Intelligence | $20.1B | $145B by 2035 | 21.85% |
| AI Voice Agents (Grand View) | $2.54B | $35.24B by 2033 | 39.0% |

**Key stats**:
- Voice agent usage grew 9x in 2025
- Production implementations grew 340% YoY across 500+ organizations
- 67% of Fortune 500 companies run production voice AI systems
- Gartner: conversational AI will reduce contact center labor costs by $80B in 2026

## Platform Ecosystem (primd's Distribution Partners)

### Tier 1: Unicorns

| Platform | Funding | Valuation | Key Metric |
|---|---|---|---|
| ElevenLabs | $500M Series D (Feb 2026) | $11B | $330M ARR, 175% YoY |
| LiveKit | $100M Series C (Jan 2026) | $1B | Powers ChatGPT Advanced Voice |
| Deepgram | $130M Series C (Jan 2026) | $1.3B | 1,300+ organizations |
| PolyAI | $86M Series D (Dec 2025) | — | 2,000+ live deployments |

### Tier 2: Growth-Stage

| Platform | Funding | Revenue | Key Metric |
|---|---|---|---|
| Bland AI | $40M Series B (Jan 2025) | $3.8M (2024) | Pre-seed to Series B in 10 months |
| Vapi | $20M Series A (Oct 2024) | $8M (Apr 2025) | Developer-first |
| Retell AI | $4.6M Seed (Aug 2024) | $40M+ ARR (Jan 2026) | 300%+ QoQ user growth |

### Tier 3: Open-Source Frameworks

| Framework | Stars | Positioning |
|---|---|---|
| Pipecat (Daily.co) | 10.1K | Most widely used OSS voice agent framework |
| LiveKit Agents | Active | Unified WebRTC + Agents, powers ChatGPT Voice |

**Why this matters for primd**: These platforms are primd's distribution channels. A Pipecat plugin puts primd in front of 10K+ developers building voice agents today.

## Target Markets

### 1. AI SDR / Sales Bots

- **Market size**: $4.12B (2025), projected $15B by 2030 (29.5% CAGR)
- **Adoption**: 81% of sales teams experimenting with or using AI
- **Key players**: 11x AI, Artisan ($25M Series A), Clay, Amplemarket
- **Pain point primd solves**: Dead air after "what's your pricing?" — the single most common failure mode in sales voice AI

### 2. Customer Support Voice Agents

- **Adoption**: 80% of businesses plan AI voice in customer service by 2026
- **Labor savings**: Up to 95% of contact center costs are labor. Even 10% automation = billions saved
- **Key players**: PolyAI (2,000+ deployments), Retell AI, Zendesk
- **Pain point primd solves**: Follow-up answers land instantly because they were prefetched during the main response

### 3. In-App Voice Copilots

- **Trend**: Shift from text-based copilots to voice-based interaction
- **Deployment**: Browser-based via WASM (no server round-trip)
- **Pain point primd solves**: Runs entirely in-browser, no network latency for retrieval

### 4. Healthcare Intake & Scheduling

- **Compliance**: HIPAA requires on-premise/edge processing in many scenarios
- **Pattern**: Highly structured conversations (name → DOB → symptoms → scheduling)
- **Pain point primd solves**: Structured conversations = high Markov prediction accuracy (70-80%+)

## Developer Sentiment

### The Latency Problem is Real

- 72% of organizations cite performance quality as the top barrier (Deepgram 2025)
- "Developers are forced to choose between 'fast and frustrating' or 'smart but slow'"
- "The single biggest mistake developers make is treating voice AI like a request-response API"
- Multiple Show HN posts targeting 500ms voice-to-voice latency
- Active HN threads about voice AI latency during load spikes

### RAG Is a Known Bottleneck

- "RAG lookups taking 400ms too long can cut off ASR, causing dead air"
- Tool usage introduces 2.3x increase in first-token response time
- Most teams use filler phrases ("Let me check that for you") as a workaround
- VoiceAgentRAG's existence as a Salesforce research paper validates this as a publishable-grade problem

### Dead Air = Death

- Callers experiencing consistent delays >900ms show higher hang-up rates
- Above 1,500ms, conversations feel broken
- Users never say "latency" — they say the agent "felt off" or "kept pausing"

## On-Device / Edge Trajectory

The edge deployment trend validates primd's WASM and single-binary strategy.

- **Moonshine v2** (Feb 2026): 107ms STT latency vs Whisper's 11,286ms. 26MB model. A step change for on-device STT.
- **whisper.cpp, llama.cpp**: Thriving ecosystems for local inference
- **Privacy regulation**: Pushing enterprises toward on-premise/on-device AI
- **Cost pressure**: Cloud inference at scale is expensive; edge reduces per-query cost

primd's WASM build makes it the natural retrieval layer for this emerging edge voice AI stack.

## Market Timing

Three things converged in 2025-2026 to create this opportunity:

1. **TTS broke under 100ms** (Cartesia 40ms, ElevenLabs Flash 75ms)
2. **STT got good in streaming mode** (Deepgram Nova-3 <200ms)
3. **LLM TTFT broke under 100ms** (Groq, Cerebras)

With every other component optimized, retrieval's 50-300ms stands out as the remaining bottleneck. Two years ago, retrieval latency was hidden by slower STT/TTS/LLM. Now it's exposed.

**Sources**: Hamming AI, Retell AI, Deepgram, MarketsAndMarkets, Grand View Research, Fortune Business Insights, TechCrunch, AssemblyAI, Vonage, VoiceAgentRAG (arXiv 2603.02206).
