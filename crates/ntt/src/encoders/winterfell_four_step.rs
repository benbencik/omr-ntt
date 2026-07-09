// Four-step transpose-based parallel NTT
// Source project: winterfell/math
// Source path: math/src/fft/concurrent.rs -- split_radix_fft (adapted from https://github.com/0xProject/OpenZKP)
// Reference: Bailey, "FFTs in External or Hierarchical Memory" (J. Supercomput., 1990)

use ark_ff::FftField;
use rayon::prelude::*;

use crate::encoder::{NttDomain, NttEncoder};

// Decomposes N-point NTT into inner_len inner FFTs and inner_len outer FFTs
// connected by two in-place matrix transposes; rows within each set are
// independent so each half parallelises with no cross-thread communication.
pub struct WinterfellFourStep;

impl<F: FftField + Send + Sync> NttEncoder<F> for WinterfellFourStep {
    #[allow(non_snake_case)]
    fn ntt_full(&self, buf: &mut [F], domain: &NttDomain<F>) {
        assert_eq!(buf.len(), domain.N);
        if domain.N > 1 {
            split_radix_fft_parallel(buf, &domain.bitrev_twiddles);
            derange(buf, domain.log_N);
        }
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
    let stretch = outer_len / inner_len; // 1 when log_n even, 2 when odd

    // g = twiddles[N/4] = omega (primitive N-th root of unity)
    let g = twiddles[twiddles.len() / 2];
    let log_inner = inner_len.ilog2();

    // Step 1: transpose inner_len × inner_len × stretch matrix
    transpose_square_stretch(values, inner_len, stretch);

    // Step 2: parallel inner FFTs (inner_len rows × outer_len each)
    // Each row: stretch interleaved inner_len-point FFTs
    values.par_chunks_mut(outer_len).for_each(|row| {
        fft_in_place(row, twiddles, stretch, stretch, 0);
    });

    // Step 3: transpose back
    transpose_square_stretch(values, inner_len, stretch);

    // Step 4: four-step twiddle multiply + parallel outer FFTs
    values
        .par_chunks_mut(outer_len)
        .enumerate()
        .for_each(|(i, row)| {
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

fn transpose_square_stretch<T>(matrix: &mut [T], size: usize, stretch: usize) {
    match stretch {
        1 => transpose_square_1(matrix, size),
        2 => transpose_square_2(matrix, size),
        _ => unreachable!("stretch must be 1 or 2"),
    }
}

fn transpose_square_1<T>(matrix: &mut [T], size: usize) {
    debug_assert_eq!(matrix.len(), size * size);
    for row in (0..size).step_by(2) {
        let i = row * size + row;
        matrix.swap(i + 1, i + size);
        for col in (row..size).step_by(2).skip(1) {
            let i = row * size + col;
            let j = col * size + row;
            matrix.swap(i, j);
            matrix.swap(i + 1, j + size);
            matrix.swap(i + size, j + 1);
            matrix.swap(i + size + 1, j + size + 1);
        }
    }
}

fn transpose_square_2<T>(matrix: &mut [T], size: usize) {
    debug_assert_eq!(matrix.len(), 2 * size * size);
    for row in 0..size {
        for col in (row..size).skip(1) {
            let i = (row * size + col) * 2;
            let j = (col * size + row) * 2;
            matrix.swap(i, j);
            matrix.swap(i + 1, j + 1);
        }
    }
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
