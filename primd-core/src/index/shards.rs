use super::events::EventCatalog;
use super::signatures::SignatureIndex;
use crate::embed::binary::BinarySignature;
use crate::predict::EventId;

#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub coarse_k: usize,
    pub max_candidate_events: usize,
    pub parallel: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            coarse_k: 64,
            max_candidate_events: 8,
            parallel: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    pub results: Vec<(u32, usize)>,
    pub coarse_results: Vec<(u32, usize)>,
    pub candidate_events: Vec<EventId>,
    pub shard_scope_size: usize,
    pub used_shards: bool,
}

pub struct HierarchicalIndex {
    signatures: SignatureIndex,
    events: EventCatalog,
}

impl HierarchicalIndex {
    pub fn new(signatures: SignatureIndex, events: EventCatalog) -> Self {
        Self { signatures, events }
    }

    pub fn signatures(&self) -> &SignatureIndex {
        &self.signatures
    }

    pub fn events(&self) -> &EventCatalog {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }

    pub fn search(
        &self,
        query: &BinarySignature,
        top_k: usize,
        options: &SearchOptions,
    ) -> SearchResult {
        if top_k == 0 || self.signatures.is_empty() {
            return SearchResult::default();
        }

        let coarse_k = options.coarse_k.max(top_k).min(self.signatures.len());
        let coarse_results = if options.parallel {
            self.signatures.scan_top_k_parallel(query, coarse_k)
        } else {
            self.signatures.scan_top_k(query, coarse_k)
        };

        if self.events.is_empty() || self.events.len() <= 1 {
            return SearchResult {
                results: coarse_results.iter().take(top_k).copied().collect(),
                coarse_results,
                candidate_events: Vec::new(),
                shard_scope_size: self.signatures.len(),
                used_shards: false,
            };
        }

        let event_seed_len = coarse_results.len().min(top_k.saturating_mul(4).max(1));
        let candidate_events = self.events.candidate_events_from_docs(
            &coarse_results[..event_seed_len],
            options.max_candidate_events.max(1),
        );
        let scope = self.events.union_scope(&candidate_events);
        if scope.is_empty() {
            return SearchResult {
                results: coarse_results.iter().take(top_k).copied().collect(),
                coarse_results,
                candidate_events,
                shard_scope_size: self.signatures.len(),
                used_shards: false,
            };
        }

        let results = if options.parallel {
            self.signatures
                .scan_top_k_subset_parallel(query, &scope, top_k)
        } else {
            self.signatures.scan_top_k_subset(query, &scope, top_k)
        };

        SearchResult {
            results,
            coarse_results,
            candidate_events,
            shard_scope_size: scope.len(),
            used_shards: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::embed::binary::BinarySignature;

    fn sig(seed: u8) -> BinarySignature {
        BinarySignature([seed; 32])
    }

    #[test]
    fn hierarchical_search_uses_event_scope() {
        let signatures = SignatureIndex::new(vec![sig(0x10), sig(0x11), sig(0xF0), sig(0xF1)]);
        let mut named = BTreeMap::new();
        named.insert("a".to_string(), vec![0, 1]);
        named.insert("b".to_string(), vec![2, 3]);
        let events = EventCatalog::from_named_scope(&named, 4);
        let index = HierarchicalIndex::new(signatures, events);

        let result = index.search(
            &sig(0x10),
            1,
            &SearchOptions {
                coarse_k: 2,
                max_candidate_events: 1,
                parallel: true,
            },
        );
        assert!(result.used_shards);
        assert_eq!(result.results[0].1, 0);
        assert!(result.shard_scope_size < index.len());
    }
}
