#![feature(portable_simd)]
#![feature(slice_as_chunks)]

use std::simd::{Simd, SimdFloat, SimdPartialOrd, LaneCount, SupportedLaneCount, Mask};
use std::sync::atomic::{AtomicU64, Ordering};
use std::ptr::NonNull;
use std::alloc::{handle_alloc_error, Layout};
use rayon::prelude::*;

/// --- MÉMOIRE ALIGNÉE POUR SIMD ---
pub struct AlignedBuffer<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    ptr: NonNull<f32>,
    len: usize,
    capacity: usize,
    layout: Layout,
}

impl<const N: usize> AlignedBuffer<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn new(len: usize, fill_value: f32) -> Self {
        let align = 64; // Optimisé cache line / AVX-512
        let capacity = (len + N - 1) & !(N - 1); // Arrondi multiple N
        let layout = Layout::from_size_align(capacity * std::mem::size_of::<f32>(), align)
            .expect("Layout invalide");

        let ptr = unsafe {
            let raw_ptr = std::alloc::alloc(layout) as *mut f32;
            if raw_ptr.is_null() { handle_alloc_error(layout); }

            for i in 0..capacity {
                raw_ptr.add(i).write(if i < len { fill_value } else { 0.0 });
            }
            NonNull::new_unchecked(raw_ptr)
        };

        Self { ptr, len, capacity, layout }
    }

    #[inline(always)]
    pub fn as_simd_slice(&self) -> &[Simd<f32, N>] {
        unsafe {
            let slice = std::slice::from_raw_parts(self.ptr.as_ptr(), self.capacity);
            let (prefix, chunks, suffix) = slice.align_to::<Simd<f32, N>>();
            debug_assert!(prefix.is_empty() && suffix.is_empty());
            chunks
        }
    }

    pub fn len(&self) -> usize { self.len }
}

impl<const N: usize> Drop for AlignedBuffer<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, self.layout) }
    }
}

unsafe impl<const N: usize> Send for AlignedBuffer<N> where LaneCount<N>: SupportedLaneCount {}
unsafe impl<const N: usize> Sync for AlignedBuffer<N> where LaneCount<N>: SupportedLaneCount {}

/// --- STRATÉGIE SIMD ---
pub trait SIMDStrategy<const N: usize>: Send + Sync
where LaneCount<N>: SupportedLaneCount
{
    fn process(&self, data: Simd<f32, N>, mask: Mask<f32, N>) -> u64;
}

/// --- MOTEUR HPC ---
pub struct ArbEngine<const N: usize, S>
where
    LaneCount<N>: SupportedLaneCount,
    S: SIMDStrategy<N>,
{
    strategy: S,
    pub hits: AtomicU64,
}

impl<const N: usize, S> ArbEngine<N, S>
where
    LaneCount<N>: SupportedLaneCount,
    S: SIMDStrategy<N>,
{
    pub fn new(strategy: S) -> Self {
        Self { strategy, hits: AtomicU64::new(0) }
    }

    pub fn run(&self, buffer: &AlignedBuffer<N>, chunk_size: usize) {
        let simd_data = buffer.as_simd_slice();
        let total_lanes = buffer.len();

        // Pré-calcul du masque du dernier vecteur
        let last_vector_valid_lanes = total_lanes % N;
        let last_mask = if last_vector_valid_lanes == 0 {
            Mask::from_array([true; N])
        } else {
            let mut bits = [false; N];
            for i in 0..last_vector_valid_lanes { bits[i] = true; }
            Mask::from_array(bits)
        };

        // Parcours parallèle
        let total_hits: u64 = simd_data
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, sub_slice)| {
                let mut local_hits = 0u64;
                let is_last_chunk = (chunk_idx + 1) * chunk_size >= simd_data.len();

                // Boucle unrolling 4x
                let (main_chunks, remainder) = sub_slice.as_chunks::<4>();
                let full_mask = Mask::from_array([true; N]);

                for [v0, v1, v2, v3] in main_chunks {
                    local_hits += self.strategy.process(*v0, full_mask);
                    local_hits += self.strategy.process(*v1, full_mask);
                    local_hits += self.strategy.process(*v2, full_mask);
                    local_hits += self.strategy.process(*v3, full_mask);
                }

                for (i, v) in remainder.iter().enumerate() {
                    let mask = if is_last_chunk && i == remainder.len() - 1 {
                        last_mask
                    } else { full_mask };
                    local_hits += self.strategy.process(*v, mask);
                }

                local_hits
            })
            .sum();

        self.hits.fetch_add(total_hits, Ordering::AcqRel);
    }
}

/// --- EXEMPLE CONCRET : DÉTECTEUR DE VOLATILITÉ ---
pub struct VolatilityDetector<const N: usize> {
    pub threshold: Simd<f32, N>,
}

impl<const N: usize> SIMDStrategy<N> for VolatilityDetector<N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<f32, N>: SimdFloat + SimdPartialOrd,
{
    #[inline(always)]
    fn process(&self, data: Simd<f32, N>, mask: Mask<f32, N>) -> u64 {
        let over_threshold = data.abs().simd_gt(self.threshold);
        (over_threshold & mask).to_bitmask().count_ones() as u64
    }
}
