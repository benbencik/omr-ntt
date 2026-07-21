use ark_ff::FftField;
use rayon::prelude::*;

use super::transpose_out_of_place::transpose_par;
use super::utils::{bitrev, inplace_radix2_dif_no_derange};
use crate::encoder::{NttDomain, NttEncoder};

pub struct TransformDecompositionV2 {
    pub s: usize,
}

impl TransformDecompositionV2 {
    pub fn new(s: usize) -> Self {
        Self { s }
    }
}

impl<F: FftField + Send + Sync> NttEncoder<F> for TransformDecompositionV2 {
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
        // output rows are left in bit-reversed order
        transposed.par_chunks_mut(n1).for_each(|row| {
            inplace_radix2_dif_no_derange(row, &inner_twiddles);
        });
        let rev_idx: Vec<usize> = (0..out_len)
            .map(|j| bitrev(j as u64, log_fft_len) as usize)
            .collect();

        // TODO: constants wrong fix
        // let num_chunks = (THREADS * 4).min(n2);
        let rows_per_chunk = 64;

        // Step 3: recombine with batched twiddle multiplication
        let acc = (0..n2)
            .into_par_iter()
            .chunks(rows_per_chunk)
            .map(|row_indices| {
                let mut local_acc = vec![F::zero(); out_len];
                let mut twiddle_exp = omega.pow([row_indices[0] as u64]);

                // iterate all rows of the chunk
                for row_idx in &row_indices {
                    // full n1 row: rev_idx can point anywhere in 0..n1
                    let row = &transposed[row_idx * n1..row_idx * n1 + n1];
                    let mut twiddle_step = F::one();

                    // add to the [out_len] partial sums in the columns accumulator
                    for (j, slot) in local_acc.iter_mut().enumerate() {
                        *slot += row[rev_idx[j]] * twiddle_step;
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
        "TransformDecompositionV2"
    }
}
