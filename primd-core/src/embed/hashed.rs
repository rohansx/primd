//! Hashed (feature-hashing) embedder.
//!
//! Tokenizes text into normalized words, produces unigram + bigram features,
//! deterministically hashes each into a fixed-dim float vector, sums, and
//! L2-normalizes. No ML dependency — but real text → real similarity, with
//! the bag-of-words bias of the simHash family.
//!
//! This is the embedder you reach for when:
//!   * You want primd's pipeline working end-to-end without bringing in
//!     Candle / ONNX / a Python sidecar.
//!   * You need deterministic, reproducible signatures across runs.
//!   * Your queries are mostly keyword-style and do not need true paraphrase
//!     understanding (a real model is better for "what does it cost" vs
//!     "how much is it"; the hashed embedder will only match the literal
//!     overlap).
//!
//! For production retrieval where paraphrase recall matters, swap this for a
//! Sentence-Transformer or BGE-small via Candle. The `Embedder` trait makes
//! that change a one-line constructor swap.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::embedder::Embedder;

const DEFAULT_HASH_BANDS: usize = 4;

/// Common English stop words. Removing them dramatically improves retrieval
/// quality on short documents where the signal-to-noise from filler words
/// otherwise dominates the cosine similarity.
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "i", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "you",
    "your", "this", "do", "does", "did", "can", "could", "would", "should", "if", "but", "not",
    "no", "my", "me", "we", "our", "us", "they", "them", "their", "he", "she", "his", "her", "him",
    "any", "all", "every", "some", "such", "than", "then", "there", "these", "those", "what",
    "when", "where", "which", "who", "why", "how", "so", "too", "very",
];

pub struct HashedEmbedder {
    dim: usize,
    hash_bands: usize,
    seed: u64,
    use_bigrams: bool,
    drop_stopwords: bool,
}

impl HashedEmbedder {
    /// `dim` should be 256 for direct quantization, or any dim that matches
    /// your PCA projector's `source_dim`. 256, 384, 512 are all reasonable.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            hash_bands: DEFAULT_HASH_BANDS,
            seed: 0xC0DE_FACE_F00D_BEEF,
            use_bigrams: true,
            drop_stopwords: true,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_hash_bands(mut self, bands: usize) -> Self {
        self.hash_bands = bands.max(1);
        self
    }

    pub fn without_bigrams(mut self) -> Self {
        self.use_bigrams = false;
        self
    }

    pub fn keep_stopwords(mut self) -> Self {
        self.drop_stopwords = false;
        self
    }

    fn token_hash(&self, token: &str, band: usize) -> u64 {
        let mut h = DefaultHasher::new();
        self.seed.hash(&mut h);
        band.hash(&mut h);
        token.hash(&mut h);
        h.finish()
    }

    /// Add a token's contribution to the accumulator. Each token contributes
    /// `hash_bands` ±1 entries, each at a deterministic position. This is the
    /// "hashing trick" that gives well-defined approximate inner products.
    fn accumulate(&self, token: &str, acc: &mut [f32]) {
        for band in 0..self.hash_bands {
            let h = self.token_hash(token, band);
            let pos = (h as usize) % self.dim;
            // Use the next bit to decide the sign — keeps E[contribution]=0.
            let sign = if (h >> 32) & 1 == 0 { 1.0 } else { -1.0 };
            acc[pos] += sign;
        }
    }
}

impl Embedder for HashedEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut tokens = tokenize(text);
        if self.drop_stopwords {
            tokens.retain(|t| !is_stopword(t));
        }
        let mut acc = vec![0.0f32; self.dim];

        for tok in &tokens {
            self.accumulate(tok, &mut acc);
        }

        if self.use_bigrams {
            for w in tokens.windows(2) {
                let bigram = format!("{}_{}", w[0], w[1]);
                self.accumulate(&bigram, &mut acc);
            }
        }

        l2_normalize(&mut acc);
        acc
    }
}

fn is_stopword(token: &str) -> bool {
    STOP_WORDS.contains(&token)
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_calls() {
        let e = HashedEmbedder::new(256);
        let a = e.embed("how much does the premium plan cost");
        let b = e.embed("how much does the premium plan cost");
        assert_eq!(a, b);
    }

    #[test]
    fn similar_texts_have_higher_cosine_than_dissimilar() {
        let e = HashedEmbedder::new(512);
        let q = e.embed("how much does the premium plan cost");
        let close = e.embed("what is the cost of premium plans");
        let far = e.embed("the weather in tokyo is sunny today");
        let close_sim = cosine_similarity(&q, &close);
        let far_sim = cosine_similarity(&q, &far);
        assert!(
            close_sim > far_sim,
            "close ({close_sim}) should beat far ({far_sim})"
        );
    }

    #[test]
    fn empty_text_returns_zero_vector() {
        let e = HashedEmbedder::new(64);
        let v = e.embed("");
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn tokenization_strips_punctuation_and_lowercases() {
        let toks = tokenize("Hello, World! 123 -- foo_bar.");
        assert_eq!(
            toks,
            vec![
                "hello".to_string(),
                "world".to_string(),
                "123".to_string(),
                "foo".to_string(),
                "bar".to_string(),
            ]
        );
    }

    #[test]
    fn bigrams_help_distinguish_word_orders() {
        // Without bigrams, "dog bites man" and "man bites dog" embed identically.
        let with_bi = HashedEmbedder::new(512);
        let no_bi = HashedEmbedder::new(512).without_bigrams();

        let a = with_bi.embed("dog bites man");
        let b = with_bi.embed("man bites dog");
        let bi_sim = cosine_similarity(&a, &b);

        let a2 = no_bi.embed("dog bites man");
        let b2 = no_bi.embed("man bites dog");
        let no_bi_sim = cosine_similarity(&a2, &b2);

        // No bigrams → identical bags → cosine = 1.0
        assert!((no_bi_sim - 1.0).abs() < 1e-3);
        // With bigrams → less similar
        assert!(bi_sim < no_bi_sim);
    }

    #[test]
    fn dim_matches_constructor() {
        for dim in [64, 128, 256, 384, 512] {
            let e = HashedEmbedder::new(dim);
            assert_eq!(e.embed("test text").len(), dim);
            assert_eq!(e.dim(), dim);
        }
    }

    #[test]
    fn embedding_is_unit_norm() {
        let e = HashedEmbedder::new(256);
        let v = e.embed("a sentence with several words in it");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm should be ~1.0, got {norm}");
    }
}
