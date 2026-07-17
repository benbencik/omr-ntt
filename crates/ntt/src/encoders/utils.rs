use ark_ff::FftField;
use rayon::prelude::*;

// Vanilla sequential iterative radix-2 NTT
// running sequential derange calls to inplace_radix2_dit are already parallel
pub(super) fn inplace_radix2_dit<F: FftField>(xi: &mut [F], twiddles: &[F], log_n: u32) {
    let n = xi.len();
    if n < 2 {
        return;
    }
    debug_assert!(n.is_power_of_two());
    debug_assert_eq!(twiddles.len(), n / 2);

    // DIT: bit-reverse the input
    derange_seq(xi, log_n);

    let mut length = 2;
    while length <= n {
        let half = length / 2;
        let step = n / length;

        for i in (0..n).step_by(length) {
            for j in 0..half {
                let top_idx = i + j;
                let bot_idx = i + j + half;

                let u = xi[top_idx];
                let v = xi[bot_idx] * twiddles[j * step];

                // add and subtract
                xi[top_idx] = u + v;
                xi[bot_idx] = u - v;
            }
        }
        length *= 2;
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
