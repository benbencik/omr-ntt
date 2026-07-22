// Partial NTT: computes only the first `2*s` coefficients of an NTT

use ark_ff::FftField;
use rayon::prelude::*;

use super::transpose_out_of_place::transpose_par;
use super::utils::inplace_radix2_dit;
use crate::encoder::{NttDomain, NttEncoder};

pub struct TransformDecomposition {
    pub s: usize,
    rows_per_chunk: usize,
}

impl TransformDecomposition {
    pub fn new(s: usize) -> Self {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        println!("TransformDecomposition: using {threads} threads");
        Self {
            s,
            rows_per_chunk: threads * 4,
        }
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

        let n1 = (2 * out_len).next_power_of_two().clamp(2, n); // subroutine fft size
        let n2 = n / n1;

        let omega = domain.omega;

        let inner_twiddles: Vec<F> = (0..n1 / 2).map(|k| domain.twiddles[k * n2]).collect();
        let log_fft_len = n1.trailing_zeros();

        // rust does not want to give uninitialized (unexpected behavior)
        // allocating large vector of 0 is expensive, bypass this with unsafe
        let mut transposed = Vec::with_capacity(n);
        unsafe {
            transposed.set_len(n);
        }

        // Step 1: transpose to compute sub-FFTs on contiguous memory
        // before: n1 rows x n2 cols, after: n2 rows x n1 cols
        transpose_par(buf, &mut transposed, n2, n1);

        // Step 2: small independent sub-FFTs of size n1 (contiguous memory)
        transposed.par_chunks_mut(n1).for_each(|row| {
            inplace_radix2_dit(row, &inner_twiddles, log_fft_len);
        });

        // Step 3: recombine with batched twiddle multiplication
        let acc = (0..n2)
            .into_par_iter()
            .chunks(self.rows_per_chunk * n1)
            .map(|row_indices| {
                let mut local_acc = vec![F::zero(); out_len];
                let mut twiddle_exp = omega.pow([row_indices[0] as u64]);

                // iterate all rows of the chunk
                for row_idx in &row_indices {
                    let row = &transposed[row_idx * n1..row_idx * n1 + out_len];
                    let mut twiddle_step = F::one();

                    // add to the [out_len] partial sums in the columns accumulator
                    for (j, slot) in local_acc.iter_mut().enumerate() {
                        *slot += row[j] * twiddle_step;
                        twiddle_step *= twiddle_exp;
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
