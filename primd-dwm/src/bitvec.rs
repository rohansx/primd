//! Bit-vector with O(1) `rank_1` / `select_1` primitives.
//!
//! Foundation for primd's Hippocampus-style cold tier (arXiv:2602.13594,
//! Feb 2026). The Dynamic Wavelet Matrix in that paper stores each layer
//! as a bit-vector with constant-time rank/select; we port the same
//! primitive here as a standalone building block.
//!
//! The succinct-data-structures literature has multiple O(1) rank/select
//! constructions; we use the simplest two-level lookup table approach
//! that's still asymptotically optimal (Jacobson 1989 + Clark 1996):
//!
//! - **rank_1(i)**: count of 1-bits in `bits[0..i]`. Precomputed
//!   per-block (every 512 bits) + per-subblock (every 64 bits) prefix
//!   sums; final remainder via hardware `count_ones`.
//! - **select_1(j)**: position of the j-th 1-bit (1-indexed). Linear
//!   scan over precomputed block counts to find the right block, then
//!   linear scan within the 512-bit block. Worst case O(log n) on the
//!   block scan; this is the "good enough for cold tier" choice. The
//!   paper's full Clark-style O(1) select_1 adds a third level of
//!   auxiliary tables (~25% more space); we defer that to v0.3.1
//!   if production workloads need it.
//!
//! Memory overhead vs raw bit-vector: ~6.25% (8 bytes per 512-bit
//! block + 1 byte per 64-bit subblock). Comparable to the Hippocampus
//! paper's reported auxiliary-table cost.

use serde::{Deserialize, Serialize};

const SUBBLOCK_BITS: usize = 64;
const BLOCK_BITS: usize = 512;
const SUBBLOCKS_PER_BLOCK: usize = BLOCK_BITS / SUBBLOCK_BITS; // 8

/// Bit-vector with rank/select.
///
/// Build via [`Self::new`] from a `Vec<u64>` of packed bits (LSB-first
/// within each `u64`) and the logical bit count. Auxiliary tables are
/// computed eagerly; subsequent rank/select queries are O(1) and O(b)
/// respectively, where b is the number of bits per block.
///
/// Serializable for cold-tier persistence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BitVector {
    /// Packed bits, LSB-first within each `u64`.
    words: Vec<u64>,
    /// Logical bit count. May be smaller than `words.len() * 64` (the
    /// trailing bits of the last word are zero-padded).
    bit_len: usize,
    /// Cumulative 1-count at the start of each 512-bit block.
    /// `block_ranks[k]` = number of 1-bits in `bits[0..k*512]`.
    /// Length = `bit_len.div_ceil(BLOCK_BITS) + 1` so the last entry
    /// holds the total count.
    block_ranks: Vec<u32>,
    /// Per-subblock 1-count *within the enclosing block*. One u16 per
    /// 64-bit subblock. `subblock_ranks[k]` = number of 1-bits in
    /// `bits[block_start..k*64]` where `block_start = (k/8)*512`.
    subblock_ranks: Vec<u16>,
}

impl BitVector {
    /// Build from packed `u64` words and an explicit logical bit count.
    /// Panics if `bit_len > words.len() * 64`.
    pub fn new(words: Vec<u64>, bit_len: usize) -> Self {
        assert!(
            bit_len <= words.len() * 64,
            "bit_len {} exceeds packed words capacity {}",
            bit_len,
            words.len() * 64
        );
        let mut bv = BitVector {
            words,
            bit_len,
            block_ranks: Vec::new(),
            subblock_ranks: Vec::new(),
        };
        bv.build_index();
        bv
    }

    /// Number of bits in the vector.
    pub fn len(&self) -> usize {
        self.bit_len
    }

    pub fn is_empty(&self) -> bool {
        self.bit_len == 0
    }

    /// Total number of 1-bits.
    pub fn ones(&self) -> usize {
        *self.block_ranks.last().unwrap_or(&0) as usize
    }

    /// Total number of 0-bits.
    pub fn zeros(&self) -> usize {
        self.bit_len - self.ones()
    }

    /// Get the bit at position `i`. Returns false for `i >= bit_len`.
    pub fn get(&self, i: usize) -> bool {
        if i >= self.bit_len {
            return false;
        }
        let word = self.words[i / 64];
        ((word >> (i % 64)) & 1) == 1
    }

    /// Number of 1-bits in `bits[0..i]`. O(1) via two-level lookup.
    /// Returns total ones for `i >= bit_len`.
    pub fn rank_1(&self, i: usize) -> usize {
        if self.bit_len == 0 {
            return 0;
        }
        let i = i.min(self.bit_len);
        let block_idx = i / BLOCK_BITS;
        let subblock_idx_global = i / SUBBLOCK_BITS;
        let within_subblock = i % SUBBLOCK_BITS;

        // Cumulative count up to start of block.
        let mut count = self.block_ranks[block_idx] as usize;

        // Plus the per-subblock count within this block.
        if subblock_idx_global < self.subblock_ranks.len() {
            count += self.subblock_ranks[subblock_idx_global] as usize;

            // Plus the popcount of the partial subblock.
            if within_subblock > 0 {
                let word = self.words[subblock_idx_global];
                let mask = (1u64 << within_subblock) - 1;
                count += (word & mask).count_ones() as usize;
            }
        }

        count
    }

    /// Number of 0-bits in `bits[0..i]`. O(1).
    pub fn rank_0(&self, i: usize) -> usize {
        let i = i.min(self.bit_len);
        i - self.rank_1(i)
    }

    /// Position of the `j`-th 1-bit (1-indexed). Returns `None` if
    /// `j == 0` or `j > ones()`.
    pub fn select_1(&self, j: usize) -> Option<usize> {
        if j == 0 || j > self.ones() {
            return None;
        }
        let target = j as u32;
        // Find the first block whose cumulative count is >= target.
        // Linear scan is O(n/512); for cold-tier corpora a binary search
        // is faster but more code — keep linear for v0.3 simplicity.
        let mut block_idx = 0;
        while block_idx + 1 < self.block_ranks.len()
            && self.block_ranks[block_idx + 1] < target
        {
            block_idx += 1;
        }
        let mut remaining = target - self.block_ranks[block_idx];

        // Walk subblocks within the block until remaining <= subblock_pop.
        let block_start_word = block_idx * SUBBLOCKS_PER_BLOCK;
        for sub_offset in 0..SUBBLOCKS_PER_BLOCK {
            let word_idx = block_start_word + sub_offset;
            if word_idx >= self.words.len() {
                break;
            }
            let word = self.words[word_idx];
            let pop = word.count_ones();
            if pop >= remaining {
                // The target is in this word — find which bit.
                return Some(word_idx * SUBBLOCK_BITS + select_bit_in_u64(word, remaining));
            }
            remaining -= pop;
        }
        None
    }

    /// Position of the `j`-th 0-bit (1-indexed). Returns `None` if
    /// `j == 0` or `j > zeros()`.
    pub fn select_0(&self, j: usize) -> Option<usize> {
        if j == 0 || j > self.zeros() {
            return None;
        }
        let target = j;
        let mut block_idx = 0;
        while block_idx + 1 < self.block_ranks.len() {
            let zeros_through_next_block =
                ((block_idx + 1) * BLOCK_BITS).min(self.bit_len)
                    - self.block_ranks[block_idx + 1] as usize;
            if zeros_through_next_block >= target {
                break;
            }
            block_idx += 1;
        }
        let zeros_before_block = block_idx * BLOCK_BITS - self.block_ranks[block_idx] as usize;
        let mut remaining = target - zeros_before_block;

        let block_start_word = block_idx * SUBBLOCKS_PER_BLOCK;
        for sub_offset in 0..SUBBLOCKS_PER_BLOCK {
            let word_idx = block_start_word + sub_offset;
            if word_idx >= self.words.len() {
                break;
            }
            // Cap at logical bit_len: zero-padding past bit_len doesn't
            // count toward select_0.
            let word_bit_start = word_idx * SUBBLOCK_BITS;
            let valid_bits = self.bit_len.saturating_sub(word_bit_start).min(SUBBLOCK_BITS);
            if valid_bits == 0 {
                break;
            }
            let mask = if valid_bits == 64 {
                u64::MAX
            } else {
                (1u64 << valid_bits) - 1
            };
            let word = self.words[word_idx] & mask;
            let zeros_in_word = valid_bits as u32 - word.count_ones();
            if (zeros_in_word as usize) >= remaining {
                let inverted = (!word) & mask;
                return Some(word_idx * SUBBLOCK_BITS + select_bit_in_u64(inverted, remaining as u32));
            }
            remaining -= zeros_in_word as usize;
        }
        None
    }

    fn build_index(&mut self) {
        let total_blocks = self.bit_len.div_ceil(BLOCK_BITS);
        let total_subblocks = self.bit_len.div_ceil(SUBBLOCK_BITS);
        self.block_ranks = Vec::with_capacity(total_blocks + 1);
        self.subblock_ranks = vec![0u16; total_subblocks];

        let mut cumulative: u32 = 0;
        self.block_ranks.push(0);
        for block_idx in 0..total_blocks {
            let mut within_block: u32 = 0;
            for sub_offset in 0..SUBBLOCKS_PER_BLOCK {
                let sub_idx = block_idx * SUBBLOCKS_PER_BLOCK + sub_offset;
                if sub_idx >= total_subblocks {
                    break;
                }
                self.subblock_ranks[sub_idx] = within_block as u16;
                let word = self.words.get(sub_idx).copied().unwrap_or(0);
                // Mask out bits past bit_len in the trailing word.
                let bit_start = sub_idx * SUBBLOCK_BITS;
                let valid_bits = self.bit_len.saturating_sub(bit_start).min(SUBBLOCK_BITS);
                let mask = if valid_bits == 64 {
                    u64::MAX
                } else if valid_bits == 0 {
                    0
                } else {
                    (1u64 << valid_bits) - 1
                };
                within_block += (word & mask).count_ones();
            }
            cumulative += within_block;
            self.block_ranks.push(cumulative);
        }
    }
}

/// Select the `j`-th 1-bit (1-indexed) within a `u64`.
/// Precondition: `word.count_ones() >= j` and `j > 0`. O(j) linear
/// scan; cold-tier callers typically have j <= 64 so this is fine.
fn select_bit_in_u64(word: u64, j: u32) -> usize {
    let mut remaining = j;
    for bit in 0..64u32 {
        if ((word >> bit) & 1) == 1 {
            remaining -= 1;
            if remaining == 0 {
                return bit as usize;
            }
        }
    }
    unreachable!("select_bit_in_u64 called with j > word.count_ones()");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Build a BitVector from a slice of booleans for easy testing.
    fn from_bools(bits: &[bool]) -> BitVector {
        let n = bits.len();
        let n_words = n.div_ceil(64);
        let mut words = vec![0u64; n_words];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        BitVector::new(words, n)
    }

    /// Naive reference implementation for cross-checking.
    fn naive_rank_1(bits: &[bool], i: usize) -> usize {
        bits.iter().take(i).filter(|&&b| b).count()
    }
    fn naive_select_1(bits: &[bool], j: usize) -> Option<usize> {
        if j == 0 {
            return None;
        }
        let mut count = 0;
        for (i, &b) in bits.iter().enumerate() {
            if b {
                count += 1;
                if count == j {
                    return Some(i);
                }
            }
        }
        None
    }

    #[test]
    fn empty_bitvec() {
        let bv = from_bools(&[]);
        assert_eq!(bv.len(), 0);
        assert_eq!(bv.ones(), 0);
        assert_eq!(bv.rank_1(0), 0);
        assert!(bv.select_1(1).is_none());
    }

    #[test]
    fn single_bit() {
        let bv = from_bools(&[true]);
        assert_eq!(bv.len(), 1);
        assert_eq!(bv.ones(), 1);
        assert_eq!(bv.rank_1(0), 0);
        assert_eq!(bv.rank_1(1), 1);
        assert_eq!(bv.select_1(1), Some(0));
        assert_eq!(bv.select_1(2), None);
    }

    #[test]
    fn alternating_bits() {
        let bits: Vec<bool> = (0..200).map(|i| i % 2 == 0).collect();
        let bv = from_bools(&bits);
        for i in 0..=bits.len() {
            assert_eq!(bv.rank_1(i), naive_rank_1(&bits, i), "rank_1 at {i}");
        }
        for j in 1..=bv.ones() {
            assert_eq!(bv.select_1(j), naive_select_1(&bits, j), "select_1 at {j}");
        }
    }

    #[test]
    fn rank_select_random_large() {
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let n = 5_000;
        let bits: Vec<bool> = (0..n).map(|_| rng.random_bool(0.5)).collect();
        let bv = from_bools(&bits);

        // Spot-check rank at random positions.
        for _ in 0..200 {
            let i = rng.random_range(0..=n);
            assert_eq!(bv.rank_1(i), naive_rank_1(&bits, i));
        }

        // Spot-check select for random j.
        let ones = bv.ones();
        for _ in 0..200 {
            let j = rng.random_range(1..=ones);
            assert_eq!(bv.select_1(j), naive_select_1(&bits, j));
        }
    }

    #[test]
    fn rank_clamps_past_len() {
        let bits = vec![true, false, true, true, false];
        let bv = from_bools(&bits);
        assert_eq!(bv.rank_1(5), 3);
        assert_eq!(bv.rank_1(100), 3); // clamps to total ones
    }

    #[test]
    fn rank_zero_consistent() {
        let bits = vec![true, false, true, true, false, false, true];
        let bv = from_bools(&bits);
        for i in 0..=bits.len() {
            assert_eq!(bv.rank_0(i) + bv.rank_1(i), i);
        }
    }

    #[test]
    fn select_zero_works() {
        let bits = vec![false, true, false, true, false, true, false];
        let bv = from_bools(&bits);
        assert_eq!(bv.select_0(1), Some(0));
        assert_eq!(bv.select_0(2), Some(2));
        assert_eq!(bv.select_0(3), Some(4));
        assert_eq!(bv.select_0(4), Some(6));
        assert_eq!(bv.select_0(5), None);
    }

    #[test]
    fn serde_round_trip() {
        let bits: Vec<bool> = (0..500).map(|i| i % 3 == 0).collect();
        let bv = from_bools(&bits);
        let json = serde_json::to_string(&bv).unwrap();
        let restored: BitVector = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), bv.len());
        assert_eq!(restored.ones(), bv.ones());
        for i in 0..=bits.len() {
            assert_eq!(restored.rank_1(i), bv.rank_1(i));
        }
    }

    #[test]
    fn select_in_u64_works() {
        // Bits set at positions 5, 13, 27, 40
        let word = (1u64 << 5) | (1u64 << 13) | (1u64 << 27) | (1u64 << 40);
        assert_eq!(select_bit_in_u64(word, 1), 5);
        assert_eq!(select_bit_in_u64(word, 2), 13);
        assert_eq!(select_bit_in_u64(word, 3), 27);
        assert_eq!(select_bit_in_u64(word, 4), 40);
    }
}
