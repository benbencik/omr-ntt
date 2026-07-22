use ark_ff::FftField;
use rayon::prelude::*;

// Per-layer contiguous twiddle table of size N1 for both DIT and DIF.
pub(super) fn layer_twiddles<F: FftField>(n1: usize, n2: usize, domain_twiddles: &[F]) -> Vec<F> {
    let mut out = Vec::with_capacity(n1 - 1);
    let mut layer = 1;
    while layer < n1 {
        let step = (n1 / (2 * layer)) * n2;
        for j in 0..layer {
            out.push(domain_twiddles[j * step]);
        }
        layer *= 2;
    }
    out
}

// Get contiguous twiddles for the current layer
#[inline]
fn layer<F>(twiddles: &[F], half: usize) -> &[F] {
    &twiddles[half - 1..2 * half - 1]
}

// Vanilla sequential iterative radix-2 NTT (DIT)
// running sequential derange calls to inplace_radix2_dit are already parallel
pub(super) fn inplace_radix2_dit<F: FftField>(xi: &mut [F], twiddles: &[F], log_n: u32) {
    let n = xi.len();
    if n < 2 {
        return;
    }
    debug_assert!(n.is_power_of_two());
    debug_assert_eq!(twiddles.len(), n - 1);

    // DIT: bit-reverse the input
    derange_seq(xi, log_n);

    let mut length = 2;
    while length <= n {
        let half = length / 2;
        let cur_tw = layer(twiddles, half);

        for block in xi.chunks_mut(length) {
            let (lo, hi) = block.split_at_mut(half);
            for j in 0..half {
                let u = lo[j];
                let v = hi[j] * cur_tw[j];

                // add and subtract
                lo[j] = u + v;
                hi[j] = u - v;
            }
        }
        length *= 2;
    }
}

// same as above but does not derange the input (DIF)
pub(super) fn inplace_radix2_dif_no_derange<F: FftField>(xi: &mut [F], twiddles: &[F]) {
    let n = xi.len();
    if n < 2 {
        return;
    }
    debug_assert!(n.is_power_of_two());
    debug_assert_eq!(twiddles.len(), n - 1);

    let mut length = n;
    while length >= 2 {
        let half = length / 2;
        let cur_tw = layer(twiddles, half);

        for block in xi.chunks_mut(length) {
            let (lo, hi) = block.split_at_mut(half);
            for j in 0..half {
                let u = lo[j];
                let v = hi[j];

                lo[j] = u + v;
                hi[j] = (u - v) * cur_tw[j];
            }
        }
        length /= 2;
    }
}

pub(super) fn derange<T: Send>(xi: &mut [T], log_len: u32) {
    let n = xi.len();
    // cast to usize to bypass compiler checks, since pointers do not implement `Sync`
    let ptr = xi.as_mut_ptr() as usize;
    // skip first (0...0) and last (1...1) elements
    (1..n as u64 - 1).into_par_iter().for_each(|idx| {
        let rev_idx = bitrev(idx, log_len) as usize;
        // swap is applied only once per pair
        if idx < rev_idx as u64 {
            // This is safe because bitrev is a bijection
            unsafe {
                let p = ptr as *mut T;
                std::ptr::swap(p.add(idx as usize), p.add(rev_idx));
            }
        }
    });
}

pub(super) fn derange_seq<T>(xi: &mut [T], log_len: u32) {
    let n = xi.len();
    // skip first (0...0) and last (1...1) elements
    for idx in 1..n - 1 {
        let rev_idx = bitrev(idx as u64, log_len) as usize;
        // swap is applied only once per pair
        if idx < rev_idx {
            xi.swap(idx, rev_idx);
        }
    }
}

#[inline]
// reverse first log_len bits of a
pub(super) fn bitrev(a: u64, log_len: u32) -> u64 {
    a.reverse_bits() >> (64 - log_len)
}
