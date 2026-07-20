// Partial NTT: computes only the first `2*s` coefficients of an NTT

use ark_ff::FftField;
use rayon::prelude::*;

use super::transpose_out_of_place::transpose_par;
use super::utils::inplace_radix2_dit;
use crate::encoder::{NttDomain, NttEncoder, powers};

const THREADS: usize = 16; // TODO: detect at runtime (perhaps in constructor)

pub struct TransformDecomposition {
    pub s: usize,
}

impl TransformDecomposition {
    pub fn new(s: usize) -> Self {
        Self { s }
    }
}

impl<F: FftField + Send + Sync> NttEncoder<F> for TransformDecomposition {
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>) {
        let n = domain.N;
        assert_eq!(buf.len(), n);

        let out_len = (2 * self.s).min(n);
        if out_len == 0 {
            return;
        }

        // TODO: for n < 2^12 run FFT subroutine
        // TODO: for s < log(n) run parallel partial DFT
        
        let n2 = (2 * out_len).next_power_of_two().clamp(2, n); // subroutine fft size
        let n1 = n / n2;

        let omega = domain.omega;
        let omega_n2 = omega.pow([n1 as u64]);
        let inner_twiddles = powers(n2 / 2, omega_n2);
        let log_fft_len = n2.trailing_zeros(); // n2 is a power of two, this determines log2

        // rust does not want to give uninitialized (unexpected behavior)
        // allocating large vector of 0 is expensive, bypass this with unsafe
        let mut transposed = Vec::with_capacity(n);
        unsafe {
            transposed.set_len(n);
        }

        // Step 1: transpose to compute sub-FFTs or contiguous memory
        transpose_par(buf, &mut transposed, n1, n2);

        // Step 2: small independent sub-FFTs of size n1
        transposed.par_chunks_mut(n2).for_each(|row| {
            inplace_radix2_dit(row, &inner_twiddles, log_fft_len);
        });

        // TODO: add some safety for small n1
        // constant 4 works the best based on benchmarks
        let rows_per_batch = n1 / (THREADS * 4);

        // Step 3: recombine with batched twiddle multiplication
        let acc = transposed
            // slice the array into batches
            .par_chunks(rows_per_batch * n2)
            .enumerate() // This gives us the batch_idx, not the row_idx!
            .map(|(batch_idx, batch_data)| {
                let mut local_acc = vec![F::zero(); out_len];

                // starting row index for this specific batch
                let start_row_idx = batch_idx * rows_per_batch;

                // calculate twiddle exponent once for the entire batch
                let mut twiddle_exp = omega.pow([start_row_idx as u64]);

                // process the rows inside this batch sequentially
                for row in batch_data.chunks(n2) {
                    let mut t = F::one();
                    for (k, slot) in local_acc.iter_mut().enumerate() {
                        *slot += row[k] * t;
                        t *= twiddle_exp;
                    }
                    twiddle_exp *= omega;
                }

                local_acc
            })
            // combine all the local accumulators into the final result
            .reduce(
                || vec![F::zero(); out_len],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );

        buf[..out_len].copy_from_slice(&acc);
    }

    fn name(&self) -> &str {
        "TransformDecomposition"
    }
}
