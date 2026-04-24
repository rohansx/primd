# Prior Art

Validated citations and their relevance to primd's architecture.

## Foundational Papers

### Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.

**Relevance**: Layer 2a (binary signatures).

The canonical brain-inspired associative memory model. Operates in high-dimensional binary address space with parallel similarity search. Robust to noisy or partial cues by design — even significantly corrupted addresses retrieve correct content.

primd's binary signature scan is a direct operationalization of SDM: 256-bit binary representations, parallel hamming distance search, approximate retrieval from partial cues.

**Verified**: Foundational work, widely cited in both CS and cognitive science.

---

### Malkov, Y. & Yashunin, D. (2016). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. IEEE TPAMI.

**Relevance**: Layer 2b (per-event HNSW shards).

The current state-of-the-art for graph-based approximate nearest neighbor search. primd uses HNSW within small per-event shards rather than as a flat index over the entire corpus.

**Verified**: Standard ANN algorithm. Used by Qdrant, FAISS, Weaviate, and most vector databases.

---

### Jégou, H., Douze, M., et al. (2016). *Polysemous Codes*. ECCV.

**Relevance**: Layer 2a (binary first-stage filter).

The canonical "binary first-stage filter + asymmetric rerank" technique. Establishes that binary quantization with oversampling and full-precision rescoring recovers nearly all recall at fraction of the cost.

**Verified**: Foundational for binary quantization approaches in production (Qdrant, Elasticsearch BBQ, Weaviate).

---

## Neuroscience References

### Caucheteux, C., Gramfort, A., & King, J.-R. (2023). *Evidence of a predictive coding hierarchy in the human brain listening to speech*. Nature Human Behaviour.

**Relevance**: Layer 1 (streaming partial-query retrieval).

Showed that brain activity correlates with language model predictions at multiple timescales. Higher-order areas (including frontal regions) encode longer-range predictions, while early auditory cortex tracks immediate features.

**primd's use**: Inspired the idea of starting retrieval on partial input, with progressively refined predictions as more input arrives.

**Accuracy note**: The "8 words ahead" claim in marketing materials is a simplification of a graded, probabilistic finding. The actual result is that different cortical regions correlate with predictions at different horizons, not that the brain deterministically predicts 8 specific words.

**Verified**: Real paper, top venue. Finding is directionally correct.

---

### Zheng, J., et al. (2022). *Neurons detect cognitive boundaries to structure episodic memories in humans*. Nature Neuroscience.

**Relevance**: Layer 2 (event-segmented index).

Described neurons in the human medial temporal lobe that fire at event boundaries — transitions between distinct episodes or contexts. Recorded from epilepsy patients with implanted electrodes.

**primd's use**: Inspired the event-based document segmentation in layer 2. Documents are clustered into "events" (topically coherent groups) indexed by boundary signatures, analogous to how the hippocampus segments continuous experience into discrete retrievable episodes.

**Accuracy note**: The analogy is reasonable at a design-inspiration level. The brain's event segmentation is vastly more complex than k-means clustering. primd does not claim mechanistic equivalence.

**Verified**: Real paper, top venue. Boundary/event cell findings are accurate.

---

### Friston, K. (2010). *The free-energy principle: a unified brain theory?*. Nature Reviews Neuroscience.

**Relevance**: Layer 4 (predictive-coding delta cache).

Proposes that the brain propagates prediction errors rather than full signals. Most perception is pre-computed via top-down predictions; only the deviation (prediction error) requires bottom-up processing.

**primd's use**: Inspired the delta cache — when a new query is close to the predicted topic, primd returns the cached result adjusted by a small delta rather than performing full retrieval.

**Accuracy note**: Predictive coding is an influential theoretical framework with strong empirical support in vision, but it remains debated as a universal brain principle. It should be presented as "a leading framework that inspired our design" rather than "established fact."

**Verified**: Highly cited, influential. The theory is real but not universally accepted.

---

### Ramsauer, H., et al. (2020). *Hopfield Networks is All You Need*. ICLR 2021.

**Relevance**: General framing (attention as associative retrieval).

Proved mathematical equivalence between transformer attention mechanism and modern continuous Hopfield networks performing associative memory retrieval. Bridges classical associative memory theory with modern deep learning.

**primd's use**: Supports the general claim that retrieval can be understood through the lens of associative memory rather than database lookup.

**Verified**: Mathematical proof is sound. Published at ICLR (top ML venue).

---

## Competitor/Related Work Papers

### Salesforce AI Research (2026). *VoiceAgentRAG: Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent Architectures*. arXiv 2603.02206.

**Relevance**: Closest competitor architecture.

Dual-agent system: background "Slow Thinker" uses LLM to predict follow-up topics and prefetch into FAISS cache; foreground "Fast Talker" queries cache only. 316x speedup on cache hits, 75% overall hit rate.

**primd's differentiation**: Replaces LLM predictor with Markov matrix (20,000x faster prediction), co-locates everything in-process (no Qdrant Cloud dependency), adds streaming partial-query retrieval.

**Verified**: Real paper, real numbers. Tested on small corpus (76 chunks, 200 queries).

---

### Arora, S., et al. (2025). *StreamRAG*. arXiv 2510.02044. Meta + CMU.

**Relevance**: Overlaps with primd's Layer 1.

Speculative retrieval during speech — issues queries from partial utterances in 500ms blocks. Focuses on accuracy improvement over standard RAG, not raw latency reduction.

**primd's differentiation**: Finer-grained streaming (every 2-3 tokens, not 500ms blocks), combined with cross-turn prediction (layers 3-4) and local-first architecture.

**Verified**: Real paper, Meta Research + CMU.

---

### Qi, Y., et al. (2025). *ContextCache: Context-Aware Semantic Cache for Multi-Turn Queries*. arXiv 2506.22791. VLDB.

**Relevance**: Prior art for conversation-aware caching.

Two-stage retrieval: coarse vector filtering + fine self-attention-based contextual matching. Improves on GPTCache for multi-turn conversations.

**primd's differentiation**: primd's Layer 4 is simpler (centroid + radius, no self-attention) but faster (<0.1ms vs ~10ms). Different tradeoff: speed over accuracy for the topic-continuation case.

**Verified**: Real paper, published in VLDB.

---

## Production System References

### AWS OpenSearch (2025). Binary vectors + AVX-512 VPOPCNTDQ benchmarks.

**Relevance**: Layer 2a performance validation.

Concrete benchmarks showing 48% improvement on Sapphire Rapids with AVX-512 VPOPCNTDQ for binary vector operations. Validates primd's <0.5ms claim for 1M binary signatures.

---

### Wu, Y., et al. (2018). *The Kanerva Machine: A Generative Distributed Memory*. DeepMind.

**Relevance**: Modern revival of SDM concepts.

DeepMind's extension of Kanerva's SDM into a generative model. Validates continued relevance of SDM as a computational paradigm.

---

### Tang, H., et al. (2023). *Recurrent predictive coding models for associative memory employing covariance learning*. PLOS Computational Biology.

**Relevance**: Predictive-coding formalization applied to memory.

Formalizes the connection between predictive coding and associative memory retrieval. Theoretical support for primd's Layer 4 design.
