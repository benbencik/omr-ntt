// Radix-2 Cooley-Tukey FFT
// Source project: arkworks
// Source path: poly/src/domain/radix2_evaluation_domain.rs -- Radix2EvaluationDomain::fft_in_place

use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

use crate::encoder::{Input, NttDomain, NttEncoder};

pub struct ArkRadix2;

impl<F: FftField> NttEncoder<F> for ArkRadix2 {
    fn ntt_full(&self, input: &Input<F>, domain: &NttDomain<F>) -> Vec<F> {
        let ark_domain =
            Radix2EvaluationDomain::<F>::new(domain.N).expect("Radix2EvaluationDomain::new failed");
        let mut coeffs = input.to_dense();
        ark_domain.fft_in_place(&mut coeffs);
        coeffs
    }

    fn name(&self) -> &str {
        "ArkRadix2"
    }
}
