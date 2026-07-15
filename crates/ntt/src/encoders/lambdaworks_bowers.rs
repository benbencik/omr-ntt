//! Unchecked (tests against naive pass)
// Bowers G-network DIF NTT
// Source project: lambdaworks-math
// Source path: crates/math/src/fft/cpu/bowers_fft.rs -- bowers_fft_opt_fused, LayerTwiddles
// Reference: Bowers, "Improved Twiddle Access for Fast Fourier Transforms"

use ark_ff::FftField;

use super::utils::derange;
use crate::encoder::{NttDomain, NttEncoder};

pub struct LambdaBowers;

impl<F: FftField> NttEncoder<F> for LambdaBowers {
    #[allow(non_snake_case)]
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>) {
        let layers = layer_twiddles(domain);
        bowers_fft_opt_fused(buf, &layers);
        derange(buf, domain.log_N);
    }

    fn name(&self) -> &str {
        "LambdaBowers"
    }
}

// Layer k has N/2^(k+1) twiddles: omega^0, omega^(2^k), omega^(2*2^k), ...
// Ported from lambdaworks-math, bowers_fft.rs LayerTwiddles::new
fn layer_twiddles<F: FftField>(domain: &NttDomain<F>) -> Vec<Vec<F>> {
    let log_n = domain.log_N as usize;
    (0..log_n)
        .map(|layer| {
            let count = domain.N >> (layer + 1);
            let stride = 1usize << layer;
            (0..count).map(|i| domain.twiddles[i * stride]).collect()
        })
        .collect()
}

// DIF (decimation in frequency): output is bit-reversed; caller applies bit-reversal.
// Pairs of layers are fused to keep intermediates in registers.
// Ported from lambdaworks-math, bowers_fft.rs bowers_fft_opt_fused
fn bowers_fft_opt_fused<F: FftField>(input: &mut [F], layers: &[Vec<F>]) {
    let n = input.len();
    let log_n = n.trailing_zeros() as usize;

    if n <= 1 {
        return;
    }

    // Small sizes: fall back to single-layer processing.
    if n <= 4 {
        for layer in 0..log_n {
            let block_size = n >> layer;
            let half_block = block_size >> 1;
            for block_start in (0..n).step_by(block_size) {
                process_single_layer_block(
                    &mut input[block_start..block_start + block_size],
                    &layers[layer],
                    half_block,
                );
            }
        }
        return;
    }

    let mut layer = 0;

    // Process pairs of layers with 2-layer fusion.
    while layer + 1 < log_n {
        let block_size = n >> layer;
        if block_size >= 4 {
            let twiddles_l0 = &layers[layer];
            let twiddles_l1 = &layers[layer + 1];
            for block_start in (0..n).step_by(block_size) {
                process_fused_block(
                    &mut input[block_start..block_start + block_size],
                    twiddles_l0,
                    twiddles_l1,
                );
            }
            layer += 2;
        } else {
            break;
        }
    }

    // Handle remaining layer if log_n was odd.
    while layer < log_n {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        for block_start in (0..n).step_by(block_size) {
            process_single_layer_block(
                &mut input[block_start..block_start + block_size],
                &layers[layer],
                half_block,
            );
        }
        layer += 1;
    }
}

// At j=0: twiddles_l0[0]=1 and twiddles_l1[0]=1, saving 3 multiplications.
// Ported from lambdaworks-math, bowers_fft.rs process_fused_block
#[inline]
fn process_fused_block<F: FftField>(block: &mut [F], twiddles_l0: &[F], twiddles_l1: &[F]) {
    let quarter = block.len() >> 2;

    // j=0: twiddles_l0[0]=1 (skip w0·diff_02) and twiddles_l1[0]=1 (skip two w2 multiplies).
    {
        let w1 = twiddles_l0[quarter];
        let sum_02 = block[0] + block[2 * quarter];
        let diff_02 = block[0] - block[2 * quarter];
        let sum_13 = block[quarter] + block[3 * quarter];
        let diff_13 = block[quarter] - block[3 * quarter];
        let diff_13_w = w1 * diff_13;
        block[0] = sum_02 + sum_13;
        block[quarter] = sum_02 - sum_13;
        block[2 * quarter] = diff_02 + diff_13_w;
        block[3 * quarter] = diff_02 - diff_13_w;
    }

    for j in 1..quarter {
        let i0 = j;
        let i1 = j + quarter;
        let i2 = j + 2 * quarter;
        let i3 = j + 3 * quarter;

        let w0 = twiddles_l0[j];
        let w1 = twiddles_l0[j + quarter];
        let w2 = twiddles_l1[j];

        let sum_02 = block[i0] + block[i2];
        let diff_02 = block[i0] - block[i2];
        let diff_02_w = w0 * diff_02;

        let sum_13 = block[i1] + block[i3];
        let diff_13 = block[i1] - block[i3];
        let diff_13_w = w1 * diff_13;

        let final_0 = sum_02 + sum_13;
        let diff_sums = sum_02 - sum_13;
        let final_1 = w2 * diff_sums;

        let final_2 = diff_02_w + diff_13_w;
        let diff_diffs = diff_02_w - diff_13_w;
        let final_3 = w2 * diff_diffs;

        block[i0] = final_0;
        block[i1] = final_1;
        block[i2] = final_2;
        block[i3] = final_3;
    }
}

// At j=0: twiddles[0]=1, skip multiplication.
// Ported from lambdaworks-math, bowers_fft.rs process_single_layer_block
#[inline]
fn process_single_layer_block<F: FftField>(block: &mut [F], twiddles: &[F], half_block: usize) {
    if half_block > 0 {
        // j=0: twiddle is 1, skip multiply.
        let sum = block[0] + block[half_block];
        let diff = block[0] - block[half_block];
        block[0] = sum;
        block[half_block] = diff;
    }
    for j in 1..half_block {
        let w = twiddles[j];
        let sum = block[j] + block[j + half_block];
        let diff = block[j] - block[j + half_block];
        block[j] = sum;
        block[j + half_block] = w * diff;
    }
}
