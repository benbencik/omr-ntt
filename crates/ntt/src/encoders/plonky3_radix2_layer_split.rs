// Layer-split parallel DIT NTT (DIT first half + Bowers G^T second half)
// Source project: Plonky3
// Source path: dft/src/radix_2_dit_parallel.rs -- Radix2DitParallel::dft_batch
// Reference: Bowers, "Improved Twiddle Access for Fast Fourier Transforms"

use ark_ff::FftField;
use rayon::prelude::*;

use crate::encoder::{Input, NttDomain, NttEncoder};

// Splits log_N layers at mid = ceil(log_N / 2).
// First half: standard DIT on chunks of 2^mid (no cross-chunk sync).
// Second half: Bowers G^T on chunks of 2^(log_N-mid) (one twiddle per block).
// Three bit-reversal passes separate the halves and restore natural order.
pub struct Plonky3Radix2LayerSplit;

impl<F: FftField + Send + Sync> NttEncoder<F> for Plonky3Radix2LayerSplit {
    #[allow(non_snake_case)]
    fn ntt_full(&self, input: &Input<F>, domain: &NttDomain<F>) -> Vec<F> {
        let N = domain.N;
        let mut a = input.to_dense();
        assert_eq!(a.len(), N);
        if N <= 1 {
            return a;
        }
        layer_split_ntt(&mut a, domain);
        a
    }

    fn name(&self) -> &str {
        "Plonky3Radix2LayerSplit"
    }
}

#[allow(non_snake_case)]
fn layer_split_ntt<F: FftField + Send + Sync>(a: &mut [F], domain: &NttDomain<F>) {
    let N = domain.N;
    let log_n = domain.log_N as usize;
    let mid = (log_n + 1) / 2; // ceil(log_n / 2)

    let twiddles = &domain.twiddles;
    let bitrev_twiddles = &domain.bitrev_twiddles;

    // Step 1: bit-reverse permute
    derange(a, log_n as u32);

    // Step 2: first half — standard DIT, independent chunks of 2^mid
    // Layer k (0..mid): half_block = 2^k, twiddle for pair j = omega^{j * N/2^(k+1)}
    {
        let chunk_size = 1 << mid;
        a.par_chunks_mut(chunk_size).for_each(|chunk| {
            for layer in 0..mid {
                let half_block = 1 << layer;
                let block_size = half_block * 2;
                let twiddle_step = N >> (layer + 1);
                for block_start in (0..chunk_size).step_by(block_size) {
                    for j in 0..half_block {
                        let lo_idx = block_start + j;
                        let hi_idx = lo_idx + half_block;
                        let tw = twiddles[j * twiddle_step];
                        let lo = chunk[lo_idx];
                        let hi = chunk[hi_idx] * tw;
                        chunk[lo_idx] = lo + hi;
                        chunk[hi_idx] = lo - hi;
                    }
                }
            }
        });
    }

    // Step 3: second bit-reverse
    derange(a, log_n as u32);

    // Step 4: second half — Bowers G^T, independent chunks of 2^(log_n-mid)
    // Layer l (mid..log_n): layer_rev = log_n-1-l, half_block = 2^layer_rev
    // Each block uses ONE twiddle: bitrev_twiddles[global_block_index]
    {
        let chunk_size_2 = 1 << (log_n - mid);
        a.par_chunks_mut(chunk_size_2).enumerate().for_each(|(thread, chunk)| {
            for layer in mid..log_n {
                let first_block = thread << (layer - mid);
                let layer_rev = log_n - 1 - layer;
                let half_block = 1 << layer_rev;
                let block_size = half_block * 2;
                for (b, block) in chunk.chunks_mut(block_size).enumerate() {
                    let tw = bitrev_twiddles[first_block + b];
                    let (lo, hi) = block.split_at_mut(half_block);
                    for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
                        let lo_val = *l;
                        let hi_val = *h * tw;
                        *l = lo_val + hi_val;
                        *h = lo_val - hi_val;
                    }
                }
            }
        });
    }

    // Step 5: final bit-reverse to restore natural order
    derange(a, log_n as u32);
}

fn derange<T>(xi: &mut [T], log_len: u32) {
    for idx in 1..(xi.len() as u64 - 1) {
        let ridx = bitrev(idx, log_len);
        if idx < ridx {
            xi.swap(idx as usize, ridx as usize);
        }
    }
}

#[inline]
fn bitrev(a: u64, log_len: u32) -> u64 {
    a.reverse_bits().wrapping_shr(64 - log_len)
}
