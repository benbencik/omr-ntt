// Naive direct-evaluation NTT

use ark_ff::FftField;

use crate::encoder::{Input, NttDomain, NttEncoder};

// Computes W[j] by brute-force
// O(s*N) used for testing correctness
pub struct Naive;

impl<F: FftField> NttEncoder<F> for Naive {
    fn ntt_full(&self, input: &Input<F>, domain: &NttDomain<F>) -> Vec<F> {
        let n = domain.N;
        let entries = input.to_indexed_entries();
        let mut out = vec![F::zero(); n];
        for j in 0..n {
            for &(i, v) in &entries {
                let exp = (i * j % n) as u64; // i,j < N <= 2^32 into 64-bit
                out[j] += v * domain.omega.pow([exp]);
            }
        }
        out
    }

    fn name(&self) -> &str {
        "Naive"
    }
}
