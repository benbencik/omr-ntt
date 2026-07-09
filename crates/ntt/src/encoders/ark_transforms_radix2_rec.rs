// Recursive out-of-place radix-2 DIT NTT
// Source project: ark-transforms
// Source path: crates/transforms/src/lib.rs -- ntt_rec, ScalarCombine, ntt_with_twiddles
// Reference: Cooley & Tukey, "An Algorithm for the Machine Calculation of Complex Fourier Series"
//            (Math. Comput., 1965)

use ark_ff::FftField;
use rayon::prelude::*;

use crate::encoder::{NttDomain, NttEncoder};

// Parallel threshold: below this size recurse sequentially to avoid rayon overhead.
const PAR_THRESHOLD: usize = 1 << 12;

// Out-of-place recursive even/odd split; no bit-reversal permutation needed.
// Allocates O(N log N) memory total across all recursion levels.
pub struct ArkRadix2Rec;

impl<F: FftField + Send + Sync> NttEncoder<F> for ArkRadix2Rec {
    fn ntt_full(&self, buf: &mut [F], domain: &NttDomain<F>) {
        assert_eq!(buf.len(), domain.N);
        let out = ntt_rec(buf, &domain.twiddles);
        buf.copy_from_slice(&out);
    }

    fn name(&self) -> &str {
        "ArkRadix2Rec"
    }
}

// twiddles[j] = omega_n^j for j in 0..n/2.
// Sub-twiddles for size n/2 are every-other element: omega_{n/2}^k = omega_n^{2k}.
fn ntt_rec<F: FftField + Send + Sync>(input: &[F], twiddles: &[F]) -> Vec<F> {
    let n = input.len();
    debug_assert!(n.is_power_of_two());
    if n == 1 {
        return vec![input[0]];
    }
    debug_assert_eq!(twiddles.len(), n / 2);

    let half = n / 2;

    let mut even = Vec::with_capacity(half);
    let mut odd = Vec::with_capacity(half);
    for i in 0..half {
        even.push(input[2 * i]);
        odd.push(input[2 * i + 1]);
    }

    let sub_twiddles: Vec<F> = twiddles.iter().step_by(2).copied().collect();

    let (even_ntt, odd_ntt) = if n > PAR_THRESHOLD {
        rayon::join(
            || ntt_rec(&even, &sub_twiddles),
            || ntt_rec(&odd, &sub_twiddles),
        )
    } else {
        (ntt_rec(&even, &sub_twiddles), ntt_rec(&odd, &sub_twiddles))
    };

    let mut out = vec![F::zero(); n];
    for j in 0..half {
        let t = twiddles[j] * odd_ntt[j];
        out[j] = even_ntt[j] + t;
        out[j + half] = even_ntt[j] - t;
    }
    out
}
