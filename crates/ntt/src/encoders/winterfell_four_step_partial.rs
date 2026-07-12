//! Unchecked (tests against naive pass)

use ark_ff::FftField;

use crate::encoder::{NttDomain, NttEncoder};

/// Sparsity parameter: retain the first `2 * S` outputs.
const S: usize = 1000;

pub struct WinterfellFourStepPartial;

impl<F: FftField> NttEncoder<F> for WinterfellFourStepPartial {
    #[allow(non_snake_case)]
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>) {
        let N = domain.N;
        let m = (2 * S).min(N); // outputs we need
        if m == 0 {
            return;
        }

        let n1 = m.next_power_of_two(); // m <= n1 <= N
        let n2 = N / n1;

        // m == 1: X[0] is just the sum of all inputs.
        if n1 == 1 {
            buf[0] = buf.iter().copied().fold(F::zero(), |a, b| a + b);
            return;
        }

        // Inner NTTs: gather each stride-n2 column into a contiguous row, transform it.
        let mut mat = vec![F::zero(); N];
        for j2 in 0..n2 {
            let row = &mut mat[j2 * n1..(j2 + 1) * n1];
            for j1 in 0..n1 {
                row[j1] = buf[n2 * j1 + j2];
            }
            radix2_dit(row, domain);
        }

        // Combine columns with the twiddle omega^{j2*k1} = (omega^{k1})^{j2}.
        for k1 in 0..m {
            let base = omega_pow(domain, k1);
            let mut w = F::one();
            let mut acc = F::zero();
            for j2 in 0..n2 {
                acc += mat[j2 * n1 + k1] * w;
                w *= base;
            }
            buf[k1] = acc;
        }
    }

    fn is_partial(&self) -> bool {
        true
    }

    fn s(&self) -> Option<usize> {
        Some(S)
    }

    fn name(&self) -> &str {
        "WinterfellFourStepPartial"
    }
}

// Radix-2 DIT on a length-m power-of-two slice (natural order). The size-m root
// omega_m^j = omega_N^{j*N/m} lives in domain.twiddles at stride N/(2*gap).
fn radix2_dit<F: FftField>(buf: &mut [F], domain: &NttDomain<F>) {
    let m = buf.len();
    if m <= 1 {
        return;
    }
    derange(buf, m.trailing_zeros());

    let mut gap = 1usize;
    while gap < m {
        let chunk_size = 2 * gap;
        let step = domain.N / chunk_size;
        for chunk in buf.chunks_mut(chunk_size) {
            let (lo, hi) = chunk.split_at_mut(gap);
            for (j, (l, h)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                let t = *h * domain.twiddles[j * step];
                let u = *l;
                *l = u + t;
                *h = u - t;
            }
        }
        gap *= 2;
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

#[inline]
#[allow(non_snake_case)]
pub fn omega_pow<F: FftField>(domain: &NttDomain<F>, exp: usize) -> F {
    let N = domain.N;
    debug_assert!(exp < N, "exp={exp} out of range for N={N}");
    let half = N / 2;
    if exp == 0 {
        F::one()
    } else if exp < half {
        domain.twiddles[exp]
    } else {
        -domain.twiddles[exp - half]
    }
}
