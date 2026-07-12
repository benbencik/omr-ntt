//! Unchecked (tests against naive pass)
// Cache-oblivious four-step NTT
// Source: oxifft/src/dft/solvers/cache_oblivious.rs (Frigo/Johnson, FFTW3)
// Ported from Complex<f64> to F: FftField.
//
// Decomposes N = N1 × N2 (N1 = 2^⌊log_N/2⌋, N2 = N/N1):
//   1. Transpose N1×N2 → N2×N1  (columns become contiguous rows)
//   2. N2 parallel N1-point DIT NTTs on rows  (column FFTs of original)
//   3. Transpose back N2×N1 → N1×N2
//   4. Twiddle multiply: data[i·N2+j] *= ω^(i·j)
//   5. N1 parallel N2-point DIT NTTs on rows
//   6. Final transpose N1×N2 → N2×N1
//
// All transposes use 64-element cache blocking (vs WinterfellFourStep's 2×2).
// No explicit bit-reversal pass on the full array; derange is only called inside
// each base DIT NTT on the small sub-arrays.

use ark_ff::FftField;
use rayon::prelude::*;

use crate::encoder::{NttDomain, NttEncoder};

const BLOCK: usize = 64;

pub struct Fft3w;

impl<F: FftField + Send + Sync> NttEncoder<F> for Fft3w {
    #[allow(non_snake_case)]
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>) {
        if domain.N >= 2 {
            four_step(buf, domain);
        }
    }

    fn name(&self) -> &str {
        "FFT3W"
    }
}

#[allow(non_snake_case)]
fn four_step<F: FftField + Send + Sync>(data: &mut [F], domain: &NttDomain<F>) {
    let N = data.len();
    debug_assert!(N.is_power_of_two());

    let log_N = N.trailing_zeros() as usize;
    let log_n1 = log_N / 2;
    let n1 = 1usize << log_n1; // ≤ sqrt(N)
    let n2 = N / n1; // ≥ n1; equals n1 iff log_N even

    // Steps 1–3: column FFTs via transpose → row FFTs → transpose back
    transpose(data, n1, n2);
    data.par_chunks_mut(n1)
        .for_each(|row| base_dit_ntt(row, domain));
    transpose(data, n2, n1);

    // Step 4: twiddle multiply  ω^(i·j) for i∈1..n1, j∈1..n2
    twiddle_multiply(data, n1, n2, domain);

    // Step 5: row FFTs
    data.par_chunks_mut(n2)
        .for_each(|row| base_dit_ntt(row, domain));

    // Step 6: final transpose → natural output order
    transpose(data, n1, n2);
}

// Twiddle multiply: data[i·n2+j] *= ω^(i·j).
// ω^i = domain.twiddles[i] (valid since i < n1 ≤ √N ≤ N/2).
// Recurrence computes ω^(i·j) without pow().
fn twiddle_multiply<F: FftField>(data: &mut [F], n1: usize, n2: usize, domain: &NttDomain<F>) {
    for i in 1..n1 {
        let tw_base = domain.twiddles[i]; // ω^i
        let mut tw = tw_base; // ω^(i·1)
        let row = i * n2;
        for j in 1..n2 {
            data[row + j] *= tw;
            tw *= tw_base; // ω^(i·(j+1))
        }
    }
}

// Radix-2 DIT NTT on an arbitrary-length power-of-2 slice.
// Uses domain.twiddles with step domain.N/(2·gap); works for any sub-size m
// because ω_m = ω^(N/m) and the sub-twiddle ω_m^j = ω^(j·N/m) =
// domain.twiddles[j · (N/m)] — same table, different stride.
fn base_dit_ntt<F: FftField>(buf: &mut [F], domain: &NttDomain<F>) {
    let m = buf.len();
    if m <= 1 {
        return;
    }
    debug_assert!(m.is_power_of_two());

    derange(buf, m.trailing_zeros());

    let mut gap = 1usize;
    while gap < m {
        let chunk_size = 2 * gap;
        let twiddle_step = domain.N / chunk_size;
        for chunk in buf.chunks_mut(chunk_size) {
            let (lo, hi) = chunk.split_at_mut(gap);
            for (j, (l, h)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                let tw = domain.twiddles[j * twiddle_step];
                let t = *h * tw;
                let u = *l;
                *l = u + t;
                *h = u - t;
            }
        }
        gap *= 2;
    }
}

// Dispatch: square → in-place blocked; rectangular → out-of-place blocked.
fn transpose<F: FftField>(data: &mut [F], rows: usize, cols: usize) {
    debug_assert_eq!(data.len(), rows * cols);
    if rows == cols {
        transpose_square(data, rows);
    } else {
        transpose_rect(data, rows, cols);
    }
}

// In-place blocked transpose of an n×n square matrix (row-major).
// Block size 64 keeps each BLOCK×BLOCK tile in L1 cache.
fn transpose_square<F: Copy>(data: &mut [F], n: usize) {
    let block = BLOCK.min(n);
    let mut bi = 0;
    while bi < n {
        let bi_end = (bi + block).min(n);
        let mut bj = bi;
        while bj < n {
            let bj_end = (bj + block).min(n);
            if bi == bj {
                for i in bi..bi_end {
                    for j in (i + 1)..bj_end {
                        data.swap(i * n + j, j * n + i);
                    }
                }
            } else {
                for i in bi..bi_end {
                    for j in bj..bj_end {
                        data.swap(i * n + j, j * n + i);
                    }
                }
            }
            bj += block;
        }
        bi += block;
    }
}

// Out-of-place blocked transpose: rows×cols row-major → cols×rows row-major.
fn transpose_rect<F: FftField + Copy>(data: &mut [F], rows: usize, cols: usize) {
    let mut temp = vec![F::zero(); rows * cols];
    let block = BLOCK.min(rows.min(cols));
    let mut bi = 0;
    while bi < rows {
        let bi_end = (bi + block).min(rows);
        let mut bj = 0;
        while bj < cols {
            let bj_end = (bj + block).min(cols);
            for i in bi..bi_end {
                for j in bj..bj_end {
                    temp[j * rows + i] = data[i * cols + j];
                }
            }
            bj += block;
        }
        bi += block;
    }
    data.copy_from_slice(&temp);
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
