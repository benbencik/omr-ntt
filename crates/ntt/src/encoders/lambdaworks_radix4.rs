// Radix-4 NR DIT NTT (natural-order input, bit-reversed output)
// Source project: lambdaworks-math
// Source path: crates/math/src/fft/cpu/fft.rs -- in_place_nr_4radix_fft
// Reference: Singleton, "An algorithm for computing the mixed radix fast Fourier transform"

use ark_ff::FftField;

use crate::encoder::{Input, NttDomain, NttEncoder};

// N must be a power of 4 (log2(N) even).
pub struct LambdaRadix4;

impl<F: FftField> NttEncoder<F> for LambdaRadix4 {
    #[allow(non_snake_case)]
    fn ntt_full(&self, input: &Input<F>, domain: &NttDomain<F>) -> Vec<F> {
        let N = domain.N;
        assert!(
            domain.log_N % 2 == 0,
            "LambdaRadix4 requires N to be a power of 4 (log₂N even), got log₂N={}",
            domain.log_N
        );
        let mut a = input.to_dense();
        assert_eq!(a.len(), N);
        in_place_nr_4radix_fft(&mut a, &domain.bitrev_twiddles);
        derange(&mut a, domain.log_N);
        a
    }

    fn name(&self) -> &str {
        "LambdaRadix4"
    }
}


// Input in natural order, output in bit-reversed order; caller applies derange.
// Twiddles must be in bit-reversed order.
// Ported from lambdaworks-math, fft.rs in_place_nr_4radix_fft
fn in_place_nr_4radix_fft<F: FftField>(input: &mut [F], twiddles: &[F]) {
    debug_assert!(input.len().is_power_of_two());
    debug_assert!(input.len().trailing_zeros() % 2 == 0);

    let mut group_count = 1usize;
    let mut group_size = input.len();

    while group_count < input.len() {
        #[allow(clippy::needless_range_loop)]
        for group in 0..group_count {
            let first_in_group = group * group_size;
            let first_in_next_group = first_in_group + group_size / 4;

            let w1 = twiddles[group];
            let w2 = twiddles[2 * group];
            let w3 = twiddles[2 * group + 1];

            for i in first_in_group..first_in_next_group {
                let j = i + group_size / 4;
                let k = i + group_size / 2;
                let l = i + 3 * group_size / 4;

                let zw1 = w1 * input[k];
                let tw1 = w1 * input[l];
                let a = w2 * (input[j] + tw1);
                let b = w3 * (input[j] - tw1);

                let x = input[i] + zw1 + a;
                let y = input[i] + zw1 - a;
                let z = input[i] - zw1 + b;
                let t = input[i] - zw1 - b;

                input[i] = x;
                input[j] = y;
                input[k] = z;
                input[l] = t;
            }
        }
        group_count *= 4;
        group_size /= 4;
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
