// Radix-2 Cooley-Tukey FFT
// Source project: arkworks
// Source path: poly/src/domain/radix2_evaluation_domain.rs -- Radix2EvaluationDomain::fft_in_place

use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

use crate::encoder::{NttDomain, NttEncoder};

pub struct ArkRadix2;

impl<F: FftField> NttEncoder<F> for ArkRadix2 {
    fn ntt_full(&self, buf: &mut [F], domain: &NttDomain<F>) {
        let ark_domain =
            Radix2EvaluationDomain::<F>::new(domain.N).expect("Radix2EvaluationDomain::new failed");
        // SAFETY: fft_in_place transforms in-place without resizing. We alias buf
        // as a Vec to satisfy the &mut Vec API, then forget the Vec so buf retains
        // sole ownership of the allocation.
        let mut v = unsafe { Vec::from_raw_parts(buf.as_mut_ptr(), buf.len(), buf.len()) };
        ark_domain.fft_in_place(&mut v);
        std::mem::forget(v);
    }

    fn name(&self) -> &str {
        "ArkRadix2"
    }
}
