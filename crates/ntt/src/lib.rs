pub mod decode;
pub mod encoder;
pub mod encoders;
pub mod fields;

pub use encoder::{Input, NttDomain, NttEncoder};
pub use fields::DefaultField;

#[cfg(test)]
mod tests {
    use ark_ff::{Field, One, UniformRand, Zero};
    use ark_std::test_rng;
    use rand::seq::index;

    use crate::{
        encoder::{Input, NttDomain, NttEncoder},
        encoders::{
            ArkRadix2, LambdaBowers, LambdaRadix4, Naive,
            Plonky3Radix2DitParallel, Plonky3Radix2LayerSplit, TfheStockhamRadix8,
            WinterfellFourStep, WinterfellSplitRadix,
        },
    };
    use crate::fields::DefaultField;

    fn gen_sparse(n: usize, s: usize, rng: &mut impl rand::Rng) -> Input<DefaultField> {
        let chosen = index::sample(rng, n, s);
        let entries: Vec<(usize, DefaultField)> = chosen
            .into_iter()
            .map(|idx| (idx, DefaultField::rand(rng)))
            .collect();
        Input::from_indexed(n, entries)
    }

    fn assert_agrees_with_naive(encoder: &impl NttEncoder<DefaultField>) {
        let cases: &[(usize, usize)] = &[(64, 5), (256, 16), (512, 24), (1024, 32)];
        let mut rng = test_rng();
        let naive = Naive;

        for &(n, s) in cases {
            let domain = NttDomain::<DefaultField>::new(n);
            let input = gen_sparse(n, s, &mut rng);

            let expected_full = naive.ntt_full(&input, &domain);
            let actual_full = encoder.ntt_full(&input, &domain);
            assert_eq!(expected_full, actual_full, "{}: ntt_full mismatch at N={n}, s={s}", encoder.name());
        }
    }

    #[test]
    fn one_coeff_at_index_0_gives_all_ones() {
        let n = 64usize;
        let domain = NttDomain::<DefaultField>::new(n);
        let mut v = vec![DefaultField::zero(); n];
        v[0] = DefaultField::one();
        let out = Naive.ntt_full(&Input::Full(v), &domain);
        for (j, &w) in out.iter().enumerate() {
            assert_eq!(w, DefaultField::one(), "W[{j}] should be 1");
        }
    }

    #[test]
    fn one_coeff_at_index_1_gives_twiddles() {
        // x = [0, 1, 0, ..., 0]  =>  W[j] = omega^{1*j} = omega^j
        // Directly verifies the twiddle exponent formula W[j] = sum_i x[i] * omega^{i*j}
        let n = 64usize;
        let domain = NttDomain::<DefaultField>::new(n);
        let mut v = vec![DefaultField::zero(); n];
        v[1] = DefaultField::one();
        let out = Naive.ntt_full(&Input::Full(v), &domain);
        for j in 0..n {
            let expected = domain.omega.pow([j as u64]);
            assert_eq!(out[j], expected, "W[{j}] != omega^{j}");
        }
    }

    #[test]
    fn ntt_linearity() {
        let n = 32usize;
        let domain = NttDomain::<DefaultField>::new(n);
        let mut rng = test_rng();
        let x: Vec<DefaultField> = (0..n).map(|_| DefaultField::rand(&mut rng)).collect();
        let y: Vec<DefaultField> = (0..n).map(|_| DefaultField::rand(&mut rng)).collect();
        let a = DefaultField::rand(&mut rng);
        let b = DefaultField::rand(&mut rng);
        let xy: Vec<_> = x.iter().zip(&y).map(|(&xi, &yi)| a * xi + b * yi).collect();
        let ntt_xy = Naive.ntt_full(&Input::Full(xy), &domain);
        let ntt_x = Naive.ntt_full(&Input::Full(x), &domain);
        let ntt_y = Naive.ntt_full(&Input::Full(y), &domain);
        let combined: Vec<_> = ntt_x.iter().zip(&ntt_y).map(|(&nx, &ny)| a * nx + b * ny).collect();
        assert_eq!(ntt_xy, combined);
    }

    #[test]
    fn sparse_and_dense_inputs_agree() {
        let n = 128usize;
        let domain = NttDomain::<DefaultField>::new(n);
        let sparse = gen_sparse(n, 10, &mut test_rng());
        let dense = Input::Full(sparse.to_dense());
        assert_eq!(Naive.ntt_full(&sparse, &domain), Naive.ntt_full(&dense, &domain));
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
            let expected_full = naive.ntt_full(&input, &domain);
            let actual_full = LambdaRadix4.ntt_full(&input, &domain);
            assert_eq!(expected_full, actual_full, "LambdaRadix4: ntt_full mismatch at N={n}");
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
            let expected = naive.ntt_full(&input, &domain);
            let actual = TfheStockhamRadix8.ntt_full(&input, &domain);
            assert_eq!(expected, actual, "TfheStockhamRadix8: mismatch at N={n}");
        }
    }
}
