use std::collections::{BTreeMap, HashMap};

use crate::predict::EventId;

#[derive(Clone, Debug, Default)]
pub struct EventCatalog {
    id_to_name: Vec<String>,
    name_to_id: BTreeMap<String, EventId>,
    scope: HashMap<EventId, Vec<usize>>,
    event_for_doc: Vec<Option<EventId>>,
}

impl EventCatalog {
    pub fn from_named_scope(scope: &BTreeMap<String, Vec<usize>>, corpus_len: usize) -> Self {
        let mut out = Self {
            id_to_name: Vec::with_capacity(scope.len()),
            name_to_id: BTreeMap::new(),
            scope: HashMap::with_capacity(scope.len()),
            event_for_doc: vec![None; corpus_len],
        };

        for (i, (name, indices)) in scope.iter().enumerate() {
            let event = EventId(i as u32);
            out.id_to_name.push(name.clone());
            out.name_to_id.insert(name.clone(), event);
            out.scope.insert(event, indices.clone());
            for &idx in indices {
                if idx < out.event_for_doc.len() {
                    out.event_for_doc[idx] = Some(event);
                }
            }
        }

        out
    }

    pub fn is_empty(&self) -> bool {
        self.scope.is_empty()
    }

    pub fn len(&self) -> usize {
        self.scope.len()
    }

    pub fn event_id(&self, name: &str) -> Option<EventId> {
        self.name_to_id.get(name).copied()
    }

    pub fn event_name(&self, event: EventId) -> Option<&str> {
        self.id_to_name.get(event.0 as usize).map(|s| s.as_str())
    }

    pub fn doc_event(&self, idx: usize) -> Option<EventId> {
        self.event_for_doc.get(idx).copied().flatten()
    }

    pub fn indices_for(&self, event: EventId) -> Option<&[usize]> {
        self.scope.get(&event).map(|v| v.as_slice())
    }

    pub fn scope_map(&self) -> &HashMap<EventId, Vec<usize>> {
        &self.scope
    }

    pub fn candidate_events_from_docs(&self, docs: &[(u32, usize)], limit: usize) -> Vec<EventId> {
        let mut out = Vec::new();
        for &(_, idx) in docs {
            if let Some(event) = self.doc_event(idx)
                && !out.contains(&event)
            {
                out.push(event);
                if out.len() >= limit {
                    break;
                }
            }
        }
        out
    }

    pub fn union_scope(&self, events: &[EventId]) -> Vec<usize> {
        let mut scope = Vec::new();
        for &event in events {
            if let Some(indices) = self.scope.get(&event) {
                scope.extend_from_slice(indices);
            }
        }
        scope.sort_unstable();
        scope.dedup();
        scope
    }

    pub fn named_scope(&self) -> BTreeMap<String, Vec<usize>> {
        let mut out = BTreeMap::new();
        for (name, event) in &self.name_to_id {
            if let Some(indices) = self.scope.get(event) {
                out.insert(name.clone(), indices.clone());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_names_and_docs() {
        let mut scope = BTreeMap::new();
        scope.insert("pricing".to_string(), vec![0, 1]);
        scope.insert("trial".to_string(), vec![2, 3]);
        let catalog = EventCatalog::from_named_scope(&scope, 4);

        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog.event_name(EventId(0)), Some("pricing"));
        assert_eq!(catalog.event_name(EventId(1)), Some("trial"));
        assert_eq!(catalog.doc_event(3), Some(EventId(1)));
    }
}
