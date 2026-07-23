// Four-step transpose-based parallel NTT - no direct call this algorithm was copied and modified
// Source project: winterfell
// License: MIT
// Reference: Bailey, "FFTs in External or Hierarchical Memory"

use ark_ff::FftField;
use rayon::prelude::*;

use super::transpose_out_of_place::transpose_par;
use super::utils::{bitrev, derange};
use crate::encoder::{NttDomain, NttEncoder};

// Decomposes N-point NTT into inner_len inner FFTs and inner_len outer FFTs
// connected by two out-of-place matrix transposes (cache-oblivious recursive);
// a scratch buffer is ping-ponged so no copy-back is needed. Rows within each
// FFT set are independent so each half parallelises with no cross-thread communication.
pub struct WinterfellFourStep;

impl<F: FftField + Send + Sync> NttEncoder<F> for WinterfellFourStep {
    #[allow(non_snake_case)]
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>) {
        split_radix_fft_parallel(buf, &domain.bitrev_twiddles);
        derange(buf, domain.log_N);
    }

    fn name(&self) -> &str {
        "WinterfellFourStep"
    }
}

fn split_radix_fft_parallel<F: FftField + Send + Sync>(values: &mut [F], twiddles: &[F]) {
    let n = values.len();

    // For n <= 2 the four-step decomposition is degenerate; fall back to sequential.
    if n <= 2 {
        fft_in_place(values, twiddles, 1, 1, 0);
        return;
    }

    let log_n = n.ilog2();
    let inner_len = 1_usize << (log_n / 2);
    let outer_len = n / inner_len;

    // g = twiddles[N/4] = omega (primitive N-th root of unity)
    let g = twiddles[twiddles.len() / 2];
    let log_inner = inner_len.ilog2();

    let mut scratch = vec![F::zero(); n];

    // Step 1: transpose inner_len * inner_len (stretch elems per cell): values -> scratch
    transpose_par(values, &mut scratch, outer_len, inner_len);

    // Step 2: parallel inner FFTs (inner_len rows * outer_len each)
    // Each row: stretch interleaved inner_len-point FFTs
    scratch.par_chunks_mut(inner_len).for_each(|row| {
        fft_in_place(row, twiddles, 1, 1, 0);
    });

    // Step 3: transpose back: scratch -> values
    transpose_par(&scratch, values, inner_len, outer_len);

    // Step 4: four-step twiddle multiply + parallel outer FFTs
    values
        .par_chunks_mut(outer_len)
        .enumerate()
        .for_each(|(i, row)| {
            // TODO: this traversal can be completely moved above
            // the transpose is already visiting all cells we can prob multiply there
            if i > 0 {
                // permute_index(inner_len, i) = bitrev(i, log_inner)
                let i_perm = bitrev(i as u64, log_inner) as usize;
                let inner_tw: F = g.pow([i_perm as u64]);
                let mut outer_tw = inner_tw;
                for element in row.iter_mut().skip(1) {
                    *element *= outer_tw;
                    outer_tw *= inner_tw;
                }
            }
            fft_in_place(row, twiddles, 1, 1, 0);
        });
}

// TODO: try higher radix and hardcoded basecases
// Recursive split-radix FFT producing bit-reversed output.
// Ported from winterfell/math, fft_inputs.rs fft_in_place (adapted from OpenZKP)
fn fft_in_place<F: FftField>(
    values: &mut [F],
    twiddles: &[F],
    count: usize,
    stride: usize,
    offset: usize,
) {
    const MAX_LOOP: usize = 256;
    let size = values.len() / stride;
    debug_assert!(size.is_power_of_two());

    if size > 2 {
        if stride == count && count < MAX_LOOP {
            fft_in_place(values, twiddles, 2 * count, 2 * stride, offset);
        } else {
            fft_in_place(values, twiddles, count, 2 * stride, offset);
            fft_in_place(values, twiddles, count, 2 * stride, offset + stride);
        }
    }

    // No-twiddle butterflies (implicit twiddle = 1)
    for i in offset..(offset + count) {
        let j = i + stride;
        let t = values[i];
        values[i] = t + values[j];
        values[j] = t - values[j];
    }

    // Twiddle butterflies: multiply upper element first, then add/sub
    let last_offset = offset + size * stride;
    for (idx, off) in (offset..last_offset)
        .step_by(2 * stride)
        .enumerate()
        .skip(1)
    {
        let tw = twiddles[idx];
        for j in off..(off + count) {
            let t = values[j];
            values[j + stride] *= tw;
            let s = values[j + stride];
            values[j] = t + s;
            values[j + stride] = t - s;
        }
    }
}
