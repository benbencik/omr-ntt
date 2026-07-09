// Recursive split-radix NTT
// Source project: winterfell/math
// Source path: math/src/fft/fft_inputs.rs -- fft_in_place (adapted from https://github.com/0xProject/OpenZKP)
// Reference: Duhamel & Hollmann, "Split-radix FFT algorithm" (Electron. Lett., 1984)

use ark_ff::FftField;

use crate::encoder::{NttDomain, NttEncoder};

pub struct WinterfellSplitRadix;

impl<F: FftField> NttEncoder<F> for WinterfellSplitRadix {
    #[allow(non_snake_case)]
    fn ntt_full(&self, buf: &mut [F], domain: &NttDomain<F>) {
        assert_eq!(buf.len(), domain.N);
        if domain.N > 1 {
            fft_in_place(buf, &domain.bitrev_twiddles, 1, 1, 0);
            derange(buf, domain.log_N);
        }
    }

    fn name(&self) -> &str {
        "WinterfellSplitRadix"
    }
}

// Twiddles must be in bit-reversed order (br[i] = omega^bitrev(i, log_N-1)).
// When stride == count and count < MAX_LOOP, butterflies are batched across columns.
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

    // No-twiddle butterflies (implicit twiddle = 1).
    for i in offset..(offset + count) {
        let j = i + stride;
        let t = values[i];
        values[i] = t + values[j];
        values[j] = t - values[j];
    }

    // Twiddle butterflies: multiply upper element first, then add/sub.
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
