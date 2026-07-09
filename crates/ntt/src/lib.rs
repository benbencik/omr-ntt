pub mod decode;
pub mod encoder;
pub mod encoders;
pub mod fields;

pub use encoder::{Input, NttDomain, NttEncoder};
pub use fields::{DefaultField, Goldilocks};

#[cfg(test)]
mod tests {
    use ark_ff::{Field, One, UniformRand, Zero};
    use ark_std::test_rng;
    use rand::seq::index;

    use crate::{
        encoder::{Input, NttDomain, NttEncoder},
        encoders::{
            ArkRadix2, LambdaBowers, LambdaRadix4, Naive, Fft3w,
            Plonky3Radix2DitParallel, Plonky3Radix2LayerSplit, TfheStockhamRadix8,
            WinterfellFourStep, WinterfellSplitRadix,
        },
    };
    use crate::fields::DefaultField;

    fn gen_sparse(n: usize, s: usize, rng: &mut impl rand::Rng) -> Input<DefaultField> {
        let chosen = index::sample(rng, n, s);
        let mut v = vec![DefaultField::zero(); n];
        for idx in chosen {
            v[idx] = DefaultField::rand(rng);
        }
        v
    }

    fn assert_agrees_with_naive(encoder: &impl NttEncoder<DefaultField>) {
        let cases: &[(usize, usize)] = &[(64, 5), (256, 16), (512, 24), (1024, 32)];
        let mut rng = test_rng();

        for &(n, s) in cases {
            let domain = NttDomain::<DefaultField>::new(n);
            let input = gen_sparse(n, s, &mut rng);

            let mut expected = input.clone();
            Naive.ntt_full(&mut expected, &domain);

            let mut actual = input;
            encoder.ntt_full(&mut actual, &domain);

            assert_eq!(expected, actual, "{}: ntt_full mismatch at N={n}, s={s}", encoder.name());
        }
    }

    #[test]
    fn one_coeff_at_index_0_gives_all_ones() {
        let n = 64usize;
        let domain = NttDomain::<DefaultField>::new(n);
        let mut v = vec![DefaultField::zero(); n];
        v[0] = DefaultField::one();
        Naive.ntt_full(&mut v, &domain);
        for (j, &w) in v.iter().enumerate() {
            assert_eq!(w, DefaultField::one(), "W[{j}] should be 1");
        }
    }

    #[test]
    fn one_coeff_at_index_1_gives_twiddles() {
        // x = [0, 1, 0, ..., 0]  =>  W[j] = omega^{1*j} = omega^j
        let n = 64usize;
        let domain = NttDomain::<DefaultField>::new(n);
        let mut v = vec![DefaultField::zero(); n];
        v[1] = DefaultField::one();
        Naive.ntt_full(&mut v, &domain);
        for j in 0..n {
            assert_eq!(v[j], domain.omega.pow([j as u64]), "W[{j}] != omega^{j}");
        }
    }

    #[test]
    fn ntt_linearity() {
        let n = 32usize;
        let domain = NttDomain::<DefaultField>::new(n);
        let mut rng = test_rng();
        let x: Input<DefaultField> = (0..n).map(|_| DefaultField::rand(&mut rng)).collect();
        let y: Input<DefaultField> = (0..n).map(|_| DefaultField::rand(&mut rng)).collect();
        let a = DefaultField::rand(&mut rng);
        let b = DefaultField::rand(&mut rng);
        let mut xy: Input<DefaultField> = x.iter().zip(&y).map(|(&xi, &yi)| a * xi + b * yi).collect();
        let mut nx = x;
        let mut ny = y;
        Naive.ntt_full(&mut xy, &domain);
        Naive.ntt_full(&mut nx, &domain);
        Naive.ntt_full(&mut ny, &domain);
        let combined: Vec<_> = nx.iter().zip(&ny).map(|(&nxi, &nyi)| a * nxi + b * nyi).collect();
        assert_eq!(xy, combined);
    }

    #[test]
    fn ark_radix2_agrees_with_naive() {
        assert_agrees_with_naive(&ArkRadix2);
    }

    #[test]
    fn lambda_bowers_agrees_with_naive() {
        assert_agrees_with_naive(&LambdaBowers);
    }

    #[test]
    fn winterfell_split_radix_agrees_with_naive() {
        assert_agrees_with_naive(&WinterfellSplitRadix);
    }

    #[test]
    fn plonky3_radix2_dit_parallel_agrees_with_naive() {
        assert_agrees_with_naive(&Plonky3Radix2DitParallel);
    }

    #[test]
    fn winterfell_four_step_agrees_with_naive() {
        assert_agrees_with_naive(&WinterfellFourStep);
    }

    #[test]
    fn fft3w_agrees_with_naive() {
        assert_agrees_with_naive(&Fft3w);
    }

    #[test]
    fn plonky3_radix2_layer_split_agrees_with_naive() {
        assert_agrees_with_naive(&Plonky3Radix2LayerSplit);
    }

    #[test]
    fn lambda_radix4_agrees_with_naive() {
        let cases: &[(usize, usize)] = &[(64, 5), (256, 16), (1024, 32)];
        let mut rng = test_rng();
        let naive = Naive;
        for &(n, s) in cases {
            let domain = NttDomain::<DefaultField>::new(n);
            let input = gen_sparse(n, s, &mut rng);
            let mut expected = input.clone();
            Naive.ntt_full(&mut expected, &domain);
            let mut actual = input;
            LambdaRadix4.ntt_full(&mut actual, &domain);
            assert_eq!(expected, actual, "LambdaRadix4: ntt_full mismatch at N={n}");
        }
    }

    // TfheStockhamRadix8 requires N to be a power of 8 (log₂N divisible by 3)
    #[test]
    fn tfhe_stockham_radix8_agrees_with_naive() {
        let cases: &[(usize, usize)] = &[(8, 3), (512, 24)];
        let mut rng = test_rng();
        let naive = Naive;
        for &(n, s) in cases {
            let domain = NttDomain::<DefaultField>::new(n);
            let input = gen_sparse(n, s.min(n / 2), &mut rng);
            let mut expected = input.clone();
            Naive.ntt_full(&mut expected, &domain);
            let mut actual = input;
            TfheStockhamRadix8.ntt_full(&mut actual, &domain);
            assert_eq!(expected, actual, "TfheStockhamRadix8: mismatch at N={n}");
        }
    }
}
