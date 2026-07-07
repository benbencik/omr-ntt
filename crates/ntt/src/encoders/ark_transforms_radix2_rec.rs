// Recursive out-of-place radix-2 DIT NTT
// Source project: ark-transforms
// Source path: crates/transforms/src/lib.rs -- ntt_rec, ScalarCombine, ntt_with_twiddles
// Reference: Cooley & Tukey, "An Algorithm for the Machine Calculation of Complex Fourier Series"
//            (Math. Comput., 1965)

use ark_ff::FftField;

use crate::encoder::{Input, NttDomain, NttEncoder};

// Out-of-place recursive even/odd split; no bit-reversal permutation needed.
// Allocates O(N log N) memory total across all recursion levels.
pub struct ArkRadix2Rec;

impl<F: FftField> NttEncoder<F> for ArkRadix2Rec {
    fn ntt_full(&self, input: &Input<F>, domain: &NttDomain<F>) -> Vec<F> {
        let a = input.to_dense();
        assert_eq!(a.len(), domain.N);
        ntt_rec(&a, &domain.twiddles)
    }

    fn name(&self) -> &str {
        "ArkRadix2Rec"
    }
}

// Ported from ark-transforms ntt_rec + ScalarCombine::combine.
// twiddles[j] = omega_n^j for j in 0..n/2.
// Sub-twiddles for size n/2 are every-other element: omega_{n/2}^k = omega_n^{2k}.
fn ntt_rec<F: FftField>(input: &[F], twiddles: &[F]) -> Vec<F> {
    let n = input.len();
    debug_assert!(n.is_power_of_two());
    if n == 1 {
        return vec![input[0]];
    }
    debug_assert_eq!(twiddles.len(), n / 2);

    let half = n / 2;

    let mut even = vec![F::zero(); half];
    let mut odd = vec![F::zero(); half];
    let mut i = 0;
    while i < half {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
        i += 1;
    }

    let mut sub_twiddles = vec![F::zero(); half / 2];
    let mut k = 0;
    while k < half / 2 {
        sub_twiddles[k] = twiddles[2 * k];
        k += 1;
    }

    let even_ntt = ntt_rec(&even, &sub_twiddles);
    let odd_ntt = ntt_rec(&odd, &sub_twiddles);

    // ScalarCombine::combine
    let mut out = vec![F::zero(); n];
    let mut j = 0;
    while j < half {
        let t = twiddles[j] * odd_ntt[j];
        out[j] = even_ntt[j] + t;
        out[j + half] = even_ntt[j] - t;
        j += 1;
    }
    out
}
