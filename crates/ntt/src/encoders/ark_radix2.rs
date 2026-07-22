// Radix-2 FFT
// Source project: arkworks - direct call
// License: Apache-2.0 OR MIT

use ark_ff::FftField;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

use crate::encoder::{NttDomain, NttEncoder};

pub struct ArkRadix2;

impl<F: FftField> NttEncoder<F> for ArkRadix2 {
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>) {
        let ark_domain =
            Radix2EvaluationDomain::<F>::new(domain.N).expect("Radix2EvaluationDomain::new failed");
        let mut v = unsafe { Vec::from_raw_parts(buf.as_mut_ptr(), buf.len(), buf.len()) };
        ark_domain.fft_in_place(&mut v);
        std::mem::forget(v);
    }

    fn name(&self) -> &str {
        "ArkRadix2"
    }
}
