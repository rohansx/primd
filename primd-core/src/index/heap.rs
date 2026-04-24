//! Bounded top-K min-heap for collecting nearest signatures by Hamming distance.

use std::collections::BinaryHeap;

/// A bounded max-heap that tracks the K smallest (distance, item) pairs.
///
/// Internally uses a max-heap so the worst (largest distance) element is at the
/// root. When full, a new candidate replaces the root only if its distance is
/// smaller.
#[derive(Debug, Clone)]
pub struct TopKHeap<T: Ord + Copy> {
    k: usize,
    heap: BinaryHeap<(u32, T)>,
}

impl<T: Ord + Copy> TopKHeap<T> {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// Insert a candidate. Only kept if it's among the K smallest distances.
    #[inline]
    pub fn push(&mut self, distance: u32, item: T) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push((distance, item));
        } else if let Some(&(worst_dist, _)) = self.heap.peek()
            && distance < worst_dist
        {
            self.heap.pop();
            self.heap.push((distance, item));
        }
    }

    /// Current worst distance in the heap (the threshold for admission).
    /// Returns `u32::MAX` if the heap is not yet full.
    #[inline]
    pub fn threshold(&self) -> u32 {
        if self.heap.len() < self.k {
            u32::MAX
        } else {
            self.heap.peek().map_or(u32::MAX, |&(d, _)| d)
        }
    }

    /// Drain into a vector sorted by ascending distance.
    pub fn into_sorted_vec(self) -> Vec<(u32, T)> {
        let mut v: Vec<_> = self.heap.into_vec();
        v.sort_unstable();
        v
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Merge another heap's contents into this one.
    pub fn merge(&mut self, other: Self) {
        for (dist, item) in other.heap {
            self.push(dist, item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_heap() {
        let heap: TopKHeap<usize> = TopKHeap::new(5);
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert_eq!(heap.threshold(), u32::MAX);
    }

    #[test]
    fn zero_k() {
        let mut heap: TopKHeap<usize> = TopKHeap::new(0);
        heap.push(10, 0);
        assert!(heap.is_empty());
    }

    #[test]
    fn under_capacity() {
        let mut heap = TopKHeap::new(5);
        heap.push(10, 0usize);
        heap.push(5, 1);
        heap.push(8, 2);
        assert_eq!(heap.len(), 3);
        assert_eq!(heap.threshold(), u32::MAX); // not full yet
    }

    #[test]
    fn at_capacity_replaces_worst() {
        let mut heap = TopKHeap::new(3);
        heap.push(10, 0usize);
        heap.push(5, 1);
        heap.push(8, 2);
        assert_eq!(heap.threshold(), 10);

        // Push something closer — should evict distance=10
        heap.push(3, 3);
        assert_eq!(heap.len(), 3);
        assert_eq!(heap.threshold(), 8);

        let sorted = heap.into_sorted_vec();
        assert_eq!(sorted, vec![(3, 3), (5, 1), (8, 2)]);
    }

    #[test]
    fn does_not_admit_worse() {
        let mut heap = TopKHeap::new(2);
        heap.push(5, 0usize);
        heap.push(3, 1);
        heap.push(10, 2); // worse than threshold=5, rejected
        assert_eq!(heap.len(), 2);

        let sorted = heap.into_sorted_vec();
        assert_eq!(sorted, vec![(3, 1), (5, 0)]);
    }

    #[test]
    fn sorted_output() {
        let mut heap = TopKHeap::new(5);
        for (i, d) in [42, 7, 99, 1, 23].iter().enumerate() {
            heap.push(*d, i);
        }
        let sorted = heap.into_sorted_vec();
        let distances: Vec<u32> = sorted.iter().map(|(d, _)| *d).collect();
        assert_eq!(distances, vec![1, 7, 23, 42, 99]);
    }

    #[test]
    fn merge_heaps() {
        let mut a = TopKHeap::new(3);
        a.push(10, 0usize);
        a.push(5, 1);
        a.push(8, 2);

        let mut b = TopKHeap::new(3);
        b.push(3, 3);
        b.push(12, 4);
        b.push(1, 5);

        a.merge(b);
        let sorted = a.into_sorted_vec();
        assert_eq!(sorted, vec![(1, 5), (3, 3), (5, 1)]);
    }
}
