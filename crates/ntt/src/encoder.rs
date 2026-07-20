use ark_ff::{FftField, Field};

/// Dense input vector for an NTT of known size N.
pub type Input<F> = Vec<F>;

/// Reusable NTT parameters
#[allow(non_snake_case)]
#[derive(Clone, Debug)]
pub struct NttDomain<F: FftField> {
    /// Transform size N (power of 2)
    pub N: usize,
    pub log_N: u32,
    /// Primitive N-th root of unity
    pub omega: F,
    /// twiddles[k] = omega^k  for k = 0..N
    pub twiddles: Vec<F>,
    /// bitrev_twiddles[i] = omega^bitrev(i, log_N-1)  for i = 0..N/2
    pub bitrev_twiddles: Vec<F>,
}

#[allow(non_snake_case)]
impl<F: FftField> NttDomain<F> {
    pub fn new(N: usize) -> Self {
        assert!(N.is_power_of_two(), "NTT size N={N} must be a power of 2");
        let log_N = N.trailing_zeros();

        // N | p-1: required for an N-th root of unity to exist in F_p
        assert!(
            log_N <= F::TWO_ADICITY,
            "N={N} requires log2(N)={log_N} <= TWO_ADICITY={} (N must divide p-1)",
            F::TWO_ADICITY,
        );

        let omega = F::get_root_of_unity(N as u64)
            .expect("root of unity must exist after TWO_ADICITY check");
        debug_assert_eq!(omega.pow([N as u64]), F::one());
        let twiddles = powers(N, omega);
        let bitrev_twiddles = {
            let log_half = log_N - 1;
            (0..N / 2)
                .map(|i| twiddles[bitrev(i as u64, log_half) as usize])
                .collect()
        };
        Self {
            N,
            log_N,
            omega,
            twiddles,
            bitrev_twiddles,
        }
    }
}

#[inline]
fn bitrev(a: u64, log_len: u32) -> u64 {
    a.reverse_bits().wrapping_shr(64 - log_len)
}

/// Compute `[base^0, base^1, ..., base^(count-1)]`
pub(crate) fn powers<F: Field>(count: usize, base: F) -> Vec<F> {
    let mut out = Vec::with_capacity(count);
    let mut cur = F::one();
    for _ in 0..count {
        out.push(cur);
        cur *= base;
    }
    out
}

pub trait NttEncoder<F: FftField>: Send + Sync {
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>);

    fn name(&self) -> &str;
}

impl<F: FftField, E: NttEncoder<F> + ?Sized> NttEncoder<F> for Box<E> {
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>) {
        (**self).ntt(buf, domain)
    }
    fn name(&self) -> &str {
        (**self).name()
    }
}
