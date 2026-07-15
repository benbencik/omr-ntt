pub mod decode;
pub mod encoder;
pub mod encoders;
pub mod fields;

pub use encoder::{Input, NttDomain, NttEncoder};
pub use fields::{DefaultField, Goldilocks};

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use crate::fields::DefaultField;
    use crate::{
        encoder::{Input, NttDomain, NttEncoder},
        encoders::{
            ArkRadix2, Fft3w, LambdaBowers, LambdaRadix4, Naive, Plonky3Radix2DitParallel,
            Plonky3Radix2LayerSplit, TfheStockhamRadix8, WinterfellFourStep,
            WinterfellFourStepPartial
        },
    };
    
    const POWERS_OF_TWO: [usize; 16] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
    const EVEN_POWERS_OF_TWO: [usize; 8] = [4, 16, 64, 256, 1024, 4096, 16384, 65536];
    const POWERS_OF_TWO_DIV_BY_3: [usize; 5] = [8, 64, 512, 4096, 32768];

    fn gen_random(n: usize, rng: &mut impl rand::Rng) -> Input<DefaultField> {
        (0..n).map(|_| DefaultField::rand(rng)).collect()
    }

    fn assert_agrees_with_ark(encoder: &impl NttEncoder<DefaultField>, sizes: &[usize]) {
        let mut rng = test_rng();

        for &n in sizes {
            let domain = NttDomain::<DefaultField>::new(n);
            let input = gen_random(n, &mut rng);

            let mut expected = input.clone();
            ArkRadix2.ntt(&mut expected, &domain);

            let mut actual = input;
            encoder.ntt(&mut actual, &domain);

            assert_eq!(
                expected,
                actual,
                "{}: ntt mismatch at N={n}",
                encoder.name()
            );
        }
    }

    fn assert_prefix_agrees_with_ark(
        encoder: &impl NttEncoder<DefaultField>,
        s: usize,
        cases: &[usize]
    ) {
        let mut rng = test_rng();

        for &n in cases {
            let m = (2 * s).min(n);
            let domain = NttDomain::<DefaultField>::new(n);
            let input = gen_random(n, &mut rng);

            let mut expected = input.clone();
            ArkRadix2.ntt(&mut expected, &domain);

            let mut actual = input;
            encoder.ntt(&mut actual, &domain);

            assert_eq!(
                &actual[..m],
                &expected[..m],
                "{}: first {m} outputs mismatch at N={n}",
                encoder.name()
            );
        }
    }

    // naive O(N^2) DFT vs arkworks radix-2
    // all other encoders are checked against arkworks
    #[test]
    fn naive_agrees_with_ark() {
        assert_agrees_with_ark(&Naive, &[64, 128, 256]);
    }

    #[test]
    fn ntt_linearity() {
        let n = 32usize;
        let domain = NttDomain::<DefaultField>::new(n);
        let mut rng = test_rng();
        let x = gen_random(n, &mut rng);
        let y = gen_random(n, &mut rng);
        let a = DefaultField::rand(&mut rng);
        let b = DefaultField::rand(&mut rng);
        let mut xy: Input<DefaultField> =
            x.iter().zip(&y).map(|(&xi, &yi)| a * xi + b * yi).collect();
        let mut nx = x;
        let mut ny = y;
        Naive.ntt(&mut xy, &domain);
        Naive.ntt(&mut nx, &domain);
        Naive.ntt(&mut ny, &domain);
        let combined: Vec<_> = nx
            .iter()
            .zip(&ny)
            .map(|(&nxi, &nyi)| a * nxi + b * nyi)
            .collect();
        assert_eq!(xy, combined);
    }

    #[test]
    fn lambda_bowers_agrees_with_ark() {
        assert_agrees_with_ark(&LambdaBowers, &POWERS_OF_TWO);
    }

    #[test]
    fn plonky3_radix2_dit_parallel_agrees_with_ark() {
        assert_agrees_with_ark(&Plonky3Radix2DitParallel, &POWERS_OF_TWO);
    }

    #[test]
    fn winterfell_four_step_agrees_with_ark() {
        assert_agrees_with_ark(&WinterfellFourStep, &POWERS_OF_TWO);
    }

    #[test]
    fn fft3w_agrees_with_ark() {
        assert_agrees_with_ark(&Fft3w, &POWERS_OF_TWO);
    }

    #[test]
    fn plonky3_radix2_layer_split_agrees_with_ark() {
        assert_agrees_with_ark(&Plonky3Radix2LayerSplit, &POWERS_OF_TWO);
    }

    // LambdaRadix4 requires N to be a power of 4 (logN even)
    #[test]
    fn lambda_radix4_agrees_with_ark() {
        assert_agrees_with_ark(&LambdaRadix4, &EVEN_POWERS_OF_TWO);
    }

    // TfheStockhamRadix8 requires N to be a power of 8 (logN divisible by 3)
    #[test]
    fn tfhe_stockham_radix8_agrees_with_ark() {
        assert_agrees_with_ark(&TfheStockhamRadix8, &POWERS_OF_TWO_DIV_BY_3);
    }

    #[test]
    fn winterfell_four_step_partial_agrees_with_ark() {
        for s in [4, 16, 50] {
            assert_prefix_agrees_with_ark(&WinterfellFourStepPartial::new(s), s, &POWERS_OF_TWO);
        }
    }
}
