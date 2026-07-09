// Naive direct-evaluation NTT

use ark_ff::FftField;

use crate::encoder::{NttDomain, NttEncoder};

// Computes W[j] by brute-force
// O(s*N) used for testing correctness
pub struct Naive;

impl<F: FftField> NttEncoder<F> for Naive {
    fn ntt_full(&self, buf: &mut [F], domain: &NttDomain<F>) {
        let n = domain.N;
        let input: Vec<F> = buf.to_vec();
        for j in 0..n {
            buf[j] = F::zero();
            for (i, &v) in input.iter().enumerate() {
                let exp = (i * j % n) as u64;
                buf[j] += v * domain.omega.pow([exp]);
            }
        }
    }

    fn name(&self) -> &str {
        "Naive"
    }
}
