// Stockham auto-sort radix-8 DIT NTT (out-of-place, no bit-reversal)
// Source project: tfhe-fft
// Source path: src/dit8.rs -- stockham_core_generic, last_butterfly
// Reference: OTFFT by Takuya OKAHISA (http://wwwa.pikara.ne.jp/okojisan/otfft-en/)
//
// Requires N to be a power of 8 (log₂N divisible by 3).
// The Stockham form alternates between two buffers each pass, eliminating the
// bit-reversal permutation entirely. Each pass reduces the stride by 8×,
// so log₈(N) passes suffice.

use ark_ff::FftField;

use crate::encoder::{Input, NttDomain, NttEncoder};

pub struct TfheStockhamRadix8;

impl<F: FftField> NttEncoder<F> for TfheStockhamRadix8 {
    fn ntt_full(&self, input: &Input<F>, domain: &NttDomain<F>) -> Vec<F> {
        assert!(
            domain.log_N % 3 == 0,
            "TfheStockhamRadix8 requires N to be a power of 8 (log₂N divisible by 3), got log₂N={}",
            domain.log_N
        );
        let n = domain.N;
        let a = input.to_dense();
        assert_eq!(a.len(), n);
        ntt_stockham_r8(&a, domain)
    }

    fn name(&self) -> &str {
        "TfheStockhamRadix8"
    }
}

// Returns omega^exp given twiddles[k]=omega^k for k in 0..N/2.
// exp must be in 0..N.
#[inline(always)]
fn omega_pow<F: FftField>(twiddles: &[F], n: usize, exp: usize) -> F {
    let half = n / 2;
    if exp == 0 {
        return F::one();
    }
    if exp < half {
        twiddles[exp]
    } else {
        // omega^exp = -omega^{exp - N/2}  (since omega^{N/2} = -1)
        -twiddles[exp - half]
    }
}

// Radix-8 DFT butterfly on 8 pre-twiddled inputs.
//
// imag: omega^{3N/4} — field analogue of the complex imaginary unit i
//       (satisfies imag^2 = -1)
// w8:   omega^{N/8}  — field analogue of e^{-iπ/4}
// v8:   omega^{7N/8} — field analogue of e^{+iπ/4} (= w8^{-1})
//
// Outputs [DFT₈[0], DFT₈[1], ..., DFT₈[7]] in natural order.
#[inline(always)]
fn butterfly8<F: FftField>(y: [F; 8], imag: F, w8: F, v8: F) -> [F; 8] {
    let [y0, y1, y2, y3, y4, y5, y6, y7] = y;

    let a04 = y0 + y4;
    let s04 = y0 - y4;
    let a26 = y2 + y6;
    let i_s26 = imag * (y2 - y6);
    let a15 = y1 + y5;
    let s15 = y1 - y5;
    let a37 = y3 + y7;
    let i_s37 = imag * (y3 - y7);

    let a04_p_a26 = a04 + a26;
    let a15_p_a37 = a15 + a37;
    let s04_m_is26 = s04 - i_s26;
    let s15_m_is37 = s15 - i_s37;
    let a04_m_a26 = a04 - a26;
    let i_a15_m_a37 = imag * (a15 - a37);
    let s04_p_is26 = s04 + i_s26;
    let s15_p_is37 = s15 + i_s37;

    [
        a04_p_a26 + a15_p_a37,             // DFT[0]
        s04_m_is26 + w8 * s15_m_is37,      // DFT[1]
        a04_m_a26 - i_a15_m_a37,           // DFT[2]
        s04_p_is26 - v8 * s15_p_is37,      // DFT[3]
        a04_p_a26 - a15_p_a37,             // DFT[4]
        s04_m_is26 - w8 * s15_m_is37,      // DFT[5]
        a04_m_a26 + i_a15_m_a37,           // DFT[6]
        s04_p_is26 + v8 * s15_p_is37,      // DFT[7]
    ]
}

fn ntt_stockham_r8<F: FftField>(input: &[F], domain: &NttDomain<F>) -> Vec<F> {
    let n = domain.N;

    // Field constants for the 8-point butterfly.
    // imag = omega^{3N/4} = -omega^{N/4}, satisfying imag^2 = -1.
    // w8   = omega^{N/8},  field analogue of e^{-iπ/4} (forward DFT 8th root).
    // v8   = omega^{7N/8} = -omega^{3N/8}, the inverse: w8 * v8 = 1.
    let imag = -domain.twiddles[n / 4];
    let w8 = domain.twiddles[n / 8];
    let v8 = -domain.twiddles[3 * n / 8];

    let mut src = input.to_vec();
    let mut dst = vec![F::zero(); n];

    // Passes: s = N/8, N/64, ..., 1  (stride shrinks by 8× each pass).
    // Pass with stride s:
    //   - N/(8s) independent groups, each processing 8 sub-elements.
    //   - Group p: reads src[8sp + ks + j] for k=0..7, j=0..s.
    //   - Twiddle for element k in group p: omega^{k·p·s}.
    //   - Writes to dst[k·(N/8) + sp + j] for k=0..7.
    let mut s = n / 8;
    while s >= 1 {
        let num_groups = n / (8 * s);
        for p in 0..num_groups {
            let tw_base = p * s; // omega^{k · tw_base} is the twiddle for element k
            let tw = [
                F::one(),
                omega_pow(&domain.twiddles, n, tw_base),
                omega_pow(&domain.twiddles, n, 2 * tw_base),
                omega_pow(&domain.twiddles, n, 3 * tw_base),
                omega_pow(&domain.twiddles, n, 4 * tw_base),
                omega_pow(&domain.twiddles, n, 5 * tw_base),
                omega_pow(&domain.twiddles, n, 6 * tw_base),
                omega_pow(&domain.twiddles, n, 7 * tw_base),
            ];
            for j in 0..s {
                let base = 8 * s * p + j;
                let y = [
                    tw[0] * src[base],
                    tw[1] * src[base + s],
                    tw[2] * src[base + 2 * s],
                    tw[3] * src[base + 3 * s],
                    tw[4] * src[base + 4 * s],
                    tw[5] * src[base + 5 * s],
                    tw[6] * src[base + 6 * s],
                    tw[7] * src[base + 7 * s],
                ];
                let out = butterfly8(y, imag, w8, v8);
                let out_base = s * p + j;
                for k in 0..8 {
                    dst[k * (n / 8) + out_base] = out[k];
                }
            }
        }
        core::mem::swap(&mut src, &mut dst);
        s /= 8;
    }

    // After each pass we swap; result ends up in src.
    src
}
