//! Unchecked (tests against naive pass)
// Parallel radix-2 DIT NTT
// Source project: Plonky3
// Source path: dft/src/radix_2_dit_parallel.rs -- Plonky3Radix2DitParallel (simplified scalar port;

use ark_ff::FftField;
use rayon::prelude::*;

use super::utils::derange;
use crate::encoder::{NttDomain, NttEncoder};

// Each butterfly stage uses par_chunks_mut (rayon); chunks at a given stage are independent.
pub struct Plonky3Radix2DitParallel;

impl<F: FftField + Send + Sync> NttEncoder<F> for Plonky3Radix2DitParallel {
    #[allow(non_snake_case)]
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>) {
        ntt_in_place_parallel(buf, domain);
    }

    fn name(&self) -> &str {
        "Plonky3Radix2DitParallel"
    }
}

fn ntt_in_place_parallel<F: FftField + Send + Sync>(a: &mut [F], domain: &NttDomain<F>) {
    // Bit-reversal permutation (sequential; dominates only for very small N).
    derange(a, domain.log_N);

    // Butterfly stages: at gap=g all chunks of size 2g are independent.
    let n = domain.N;
    let mut gap = 1usize;
    while gap < n {
        let chunk_size = 2 * gap;
        let step = n / chunk_size;
        // All chunks use the same twiddle range twiddles[0..gap:step].
        a.par_chunks_mut(chunk_size).for_each(|chunk| {
            let (lo, hi) = chunk.split_at_mut(gap);
            lo.iter_mut()
                .zip(hi.iter_mut())
                .zip(domain.twiddles.iter().step_by(step))
                .for_each(|((lo_el, hi_el), &tw)| {
                    *hi_el *= tw;
                    let t = *lo_el - *hi_el;
                    *lo_el += *hi_el;
                    *hi_el = t;
                });
        });
        gap *= 2;
    }
}
