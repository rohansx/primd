//! Packed signature storage and SIMD-accelerated Hamming distance scanning.
//!
//! Provides `SignatureIndex` which holds 256-bit binary signatures and scans
//! them via the best available SIMD path (AVX-512 VPOPCNTDQ > AVX2 > scalar).

use std::path::Path;
use std::sync::OnceLock;

use bytemuck::cast_slice;
use memmap2::Mmap;
use rayon::prelude::*;

use crate::embed::binary::BinarySignature;
use crate::index::heap::TopKHeap;
use crate::{PrimdError, Result};

/// Detected SIMD capability level, cached at first use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimdLevel {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Scalar,
}

fn detect_simd() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512vpopcntdq")
            && std::arch::is_x86_feature_detected!("avx512vl")
        {
            return SimdLevel::Avx512;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
    }
    SimdLevel::Scalar
}

fn simd_level() -> SimdLevel {
    static LEVEL: OnceLock<SimdLevel> = OnceLock::new();
    *LEVEL.get_or_init(detect_simd)
}

/// Storage backend for signatures: either owned or memory-mapped.
enum Storage {
    Owned(Vec<BinarySignature>),
    Mapped { mmap: Mmap, len: usize },
}

/// Packed index of 256-bit binary signatures with SIMD-accelerated Hamming scan.
pub struct SignatureIndex {
    storage: Storage,
}

impl SignatureIndex {
    /// Create from an owned vector of signatures.
    pub fn new(signatures: Vec<BinarySignature>) -> Self {
        Self {
            storage: Storage::Owned(signatures),
        }
    }

    /// Memory-map a signatures file. The file must be a contiguous array of `[u8; 32]`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let metadata = file.metadata()?;
        let file_len = metadata.len() as usize;

        if file_len == 0 || !file_len.is_multiple_of(BinarySignature::BYTES) {
            return Err(PrimdError::InvalidSignatureFile(format!(
                "file size {} is not a multiple of {} bytes",
                file_len,
                BinarySignature::BYTES
            )));
        }

        let mmap = unsafe { Mmap::map(&file)? };
        let len = file_len / BinarySignature::BYTES;
        Ok(Self {
            storage: Storage::Mapped { mmap, len },
        })
    }

    /// Write all signatures to a file as a contiguous byte array.
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        let sigs = self.as_slice();
        let bytes: &[u8] = cast_slice(sigs);
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Borrow signatures as a slice regardless of storage type.
    pub fn as_slice(&self) -> &[BinarySignature] {
        match &self.storage {
            Storage::Owned(v) => v,
            Storage::Mapped { mmap, len } => {
                let bytes: &[u8] = &mmap[..];
                let sigs: &[BinarySignature] = cast_slice(bytes);
                debug_assert_eq!(sigs.len(), *len);
                sigs
            }
        }
    }

    pub fn len(&self) -> usize {
        match &self.storage {
            Storage::Owned(v) => v.len(),
            Storage::Mapped { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Scan all signatures, returning the top-K nearest by Hamming distance.
    ///
    /// Returns `(distance, doc_index)` pairs sorted by ascending distance.
    pub fn scan_top_k(&self, query: &BinarySignature, k: usize) -> Vec<(u32, usize)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let sigs = self.as_slice();
        let mut heap = TopKHeap::new(k);

        match simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx512 => unsafe { scan_avx512(query, sigs, &mut heap) },
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => unsafe { scan_avx2(query, sigs, &mut heap) },
            SimdLevel::Scalar => scan_scalar(query, sigs, &mut heap),
        }

        heap.into_sorted_vec()
    }

    /// Parallel scan using rayon. Splits work across threads, merges results.
    pub fn scan_top_k_parallel(&self, query: &BinarySignature, k: usize) -> Vec<(u32, usize)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let sigs = self.as_slice();
        let chunk_size = (sigs.len() / rayon::current_num_threads()).max(1024);

        let heap = sigs
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let offset = chunk_idx * chunk_size;
                let mut local_heap = TopKHeap::new(k);

                match simd_level() {
                    #[cfg(target_arch = "x86_64")]
                    SimdLevel::Avx512 => unsafe {
                        scan_avx512_offset(query, chunk, &mut local_heap, offset);
                    },
                    #[cfg(target_arch = "x86_64")]
                    SimdLevel::Avx2 => unsafe {
                        scan_avx2_offset(query, chunk, &mut local_heap, offset);
                    },
                    SimdLevel::Scalar => {
                        scan_scalar_offset(query, chunk, &mut local_heap, offset);
                    }
                }

                local_heap
            })
            .reduce(
                || TopKHeap::new(k),
                |mut a, b| {
                    a.merge(b);
                    a
                },
            );

        heap.into_sorted_vec()
    }
}

// ---------------------------------------------------------------------------
// Scalar scan
// ---------------------------------------------------------------------------

fn scan_scalar(query: &BinarySignature, sigs: &[BinarySignature], heap: &mut TopKHeap<usize>) {
    scan_scalar_offset(query, sigs, heap, 0);
}

fn scan_scalar_offset(
    query: &BinarySignature,
    sigs: &[BinarySignature],
    heap: &mut TopKHeap<usize>,
    offset: usize,
) {
    for (i, sig) in sigs.iter().enumerate() {
        let dist = query.hamming_distance(sig);
        if dist < heap.threshold() {
            heap.push(dist, offset + i);
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 scan — uses VPSHUFB nibble-lookup for popcount
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(query: &BinarySignature, sigs: &[BinarySignature], heap: &mut TopKHeap<usize>) {
    unsafe { scan_avx2_offset(query, sigs, heap, 0) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2_offset(
    query: &BinarySignature,
    sigs: &[BinarySignature],
    heap: &mut TopKHeap<usize>,
    offset: usize,
) {
    use std::arch::x86_64::*;

    unsafe {
        // Popcount lookup table for 4-bit nibbles
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let low_mask = _mm256_set1_epi8(0x0F);

        let q = _mm256_loadu_si256(query.0.as_ptr() as *const __m256i);

        for (i, sig) in sigs.iter().enumerate() {
            let s = _mm256_loadu_si256(sig.0.as_ptr() as *const __m256i);
            let xor = _mm256_xor_si256(q, s);

            // Nibble-level popcount via VPSHUFB
            let lo_nibbles = _mm256_and_si256(xor, low_mask);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(xor, 4), low_mask);
            let lo_popcnt = _mm256_shuffle_epi8(lookup, lo_nibbles);
            let hi_popcnt = _mm256_shuffle_epi8(lookup, hi_nibbles);
            let byte_popcnt = _mm256_add_epi8(lo_popcnt, hi_popcnt);

            // Sum all bytes: sad against zero gives u16 sums per 8-byte group
            let sad = _mm256_sad_epu8(byte_popcnt, _mm256_setzero_si256());

            // Extract and sum the four u64 lanes
            let dist = _mm256_extract_epi64(sad, 0) as u32
                + _mm256_extract_epi64(sad, 1) as u32
                + _mm256_extract_epi64(sad, 2) as u32
                + _mm256_extract_epi64(sad, 3) as u32;

            if dist < heap.threshold() {
                heap.push(dist, offset + i);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX-512 scan — uses VPOPCNTDQ for native 64-bit popcount
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq,avx512vl")]
unsafe fn scan_avx512(
    query: &BinarySignature,
    sigs: &[BinarySignature],
    heap: &mut TopKHeap<usize>,
) {
    unsafe { scan_avx512_offset(query, sigs, heap, 0) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq,avx512vl")]
unsafe fn scan_avx512_offset(
    query: &BinarySignature,
    sigs: &[BinarySignature],
    heap: &mut TopKHeap<usize>,
    offset: usize,
) {
    use std::arch::x86_64::*;

    unsafe {
        let q = _mm256_loadu_si256(query.0.as_ptr() as *const __m256i);

        for (i, sig) in sigs.iter().enumerate() {
            let s = _mm256_loadu_si256(sig.0.as_ptr() as *const __m256i);
            let xor = _mm256_xor_si256(q, s);
            let popcnt = _mm256_popcnt_epi64(xor);

            // Sum the four u64 lanes
            let dist = _mm256_extract_epi64(popcnt, 0) as u32
                + _mm256_extract_epi64(popcnt, 1) as u32
                + _mm256_extract_epi64(popcnt, 2) as u32
                + _mm256_extract_epi64(popcnt, 3) as u32;

            if dist < heap.threshold() {
                heap.push(dist, offset + i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sigs(n: usize) -> Vec<BinarySignature> {
        use rand::Rng;
        let mut rng = rand::rng();
        (0..n)
            .map(|_| {
                let mut bytes = [0u8; 32];
                rng.fill(&mut bytes);
                BinarySignature(bytes)
            })
            .collect()
    }

    #[test]
    fn scan_finds_exact_match() {
        let mut sigs = make_sigs(1000);
        let query = BinarySignature([0xAB; 32]);
        sigs[42] = query; // insert exact match

        let idx = SignatureIndex::new(sigs);
        let results = idx.scan_top_k(&query, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (0, 42)); // distance=0, index=42
    }

    #[test]
    fn scan_empty() {
        let idx = SignatureIndex::new(Vec::new());
        let query = BinarySignature([0x00; 32]);
        assert!(idx.scan_top_k(&query, 10).is_empty());
    }

    #[test]
    fn scan_k_zero() {
        let idx = SignatureIndex::new(make_sigs(100));
        let query = BinarySignature([0x00; 32]);
        assert!(idx.scan_top_k(&query, 0).is_empty());
    }

    #[test]
    fn scan_k_larger_than_corpus() {
        let sigs = make_sigs(5);
        let idx = SignatureIndex::new(sigs);
        let query = BinarySignature([0x00; 32]);
        let results = idx.scan_top_k(&query, 100);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn scan_results_sorted_ascending() {
        let idx = SignatureIndex::new(make_sigs(500));
        let query = BinarySignature([0x00; 32]);
        let results = idx.scan_top_k(&query, 10);

        for w in results.windows(2) {
            assert!(w[0].0 <= w[1].0, "results not sorted: {:?}", results);
        }
    }

    #[test]
    fn parallel_matches_sequential() {
        let sigs = make_sigs(10_000);
        let query = BinarySignature([0x55; 32]);
        let idx = SignatureIndex::new(sigs);

        let seq = idx.scan_top_k(&query, 10);
        let par = idx.scan_top_k_parallel(&query, 10);

        // Same distances (indices may differ if ties exist)
        let seq_dists: Vec<u32> = seq.iter().map(|(d, _)| *d).collect();
        let par_dists: Vec<u32> = par.iter().map(|(d, _)| *d).collect();
        assert_eq!(seq_dists, par_dists);
    }

    #[test]
    fn file_roundtrip() {
        let sigs = make_sigs(100);
        let dir = std::env::temp_dir().join("primd_test_sigs");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("signatures.bin");

        let original = SignatureIndex::new(sigs.clone());
        original.write_to_file(&path).unwrap();

        let loaded = SignatureIndex::from_file(&path).unwrap();
        assert_eq!(loaded.len(), 100);
        assert_eq!(loaded.as_slice(), sigs.as_slice());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn file_invalid_size() {
        let dir = std::env::temp_dir().join("primd_test_sigs_bad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.bin");
        std::fs::write(&path, &[0u8; 33]).unwrap(); // not a multiple of 32

        assert!(SignatureIndex::from_file(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_paths_agree() {
        // Compare all available SIMD paths against scalar reference
        let sigs = make_sigs(2000);
        let query = BinarySignature([0x33; 32]);

        let mut scalar_heap = TopKHeap::new(20);
        scan_scalar(&query, &sigs, &mut scalar_heap);
        let scalar_results = scalar_heap.into_sorted_vec();

        if std::arch::is_x86_feature_detected!("avx2") {
            let mut avx2_heap = TopKHeap::new(20);
            unsafe { scan_avx2(&query, &sigs, &mut avx2_heap) };
            let avx2_results = avx2_heap.into_sorted_vec();
            assert_eq!(scalar_results, avx2_results, "AVX2 disagrees with scalar");
        }

        if std::arch::is_x86_feature_detected!("avx512vpopcntdq")
            && std::arch::is_x86_feature_detected!("avx512vl")
        {
            let mut avx512_heap = TopKHeap::new(20);
            unsafe { scan_avx512(&query, &sigs, &mut avx512_heap) };
            let avx512_results = avx512_heap.into_sorted_vec();
            assert_eq!(
                scalar_results, avx512_results,
                "AVX-512 disagrees with scalar"
            );
        }
    }
}
