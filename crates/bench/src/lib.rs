use ark_ff::FftField;
use ntt::encoders::{
    ArkRadix2, ArkRadix2Rec, LambdaBowers, LambdaRadix4, Naive, Fft3w,
    Plonky3Radix2DitParallel, Plonky3Radix2LayerSplit, TfheStockhamRadix8,
    WinterfellFourStep, WinterfellSplitRadix,
};
use rand::{Rng, SeedableRng, seq::index};

pub use ntt::encoder::{Input, NttDomain, NttEncoder};

// - N: total messages per epoch (power of 2)
// - s: messages per receiver per epoch (for now default 1000)
// - k: always 2*s
#[allow(non_snake_case)]
#[derive(Clone, Debug)]
pub struct BenchParams {
    pub N: usize,
    pub s: usize,
    pub k: usize,
}

#[allow(non_snake_case)]
impl BenchParams {
    pub fn new(N: usize, s: usize) -> Self {
        assert!(N.is_power_of_two(), "N={N} must be a power of 2");
        assert!(s > 0 && s < N, "s={s} must satisfy 0 < s < N={N}");
        let k = 2 * s;
        assert!(k <= N, "k=2s={k} must be <= N={N}");
        Self { N, s, k }
    }

    pub fn n_sweep(log_ns: &[u32]) -> Vec<Self> {
        const S: usize = 1000;
        log_ns.iter().map(|&log_n| Self::new(1 << log_n, S)).collect()
    }

    pub fn n_iter(s_values: &[usize]) -> Vec<Self> {
        const LOG_N: u32 = 20;
        s_values.iter().map(|&s| Self::new(1 << LOG_N, s)).collect()
    }
}

pub fn gen_sparse_input<F: FftField>(params: &BenchParams, rng: &mut impl Rng) -> Input<F> {
    let chosen = index::sample(rng, params.N, params.s);
    let entries = chosen.into_iter().map(|idx| (idx, F::rand(rng))).collect();
    Input::from_indexed(params.N, entries)
}

pub fn gen_sparse_input_seeded<F: FftField>(params: &BenchParams, seed: u64) -> Input<F> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    gen_sparse_input(params, &mut rng)
}

pub fn all_implemented_encoders<F: FftField + Send + Sync>() -> Vec<Box<dyn NttEncoder<F>>> {
    vec![
        Box::new(Naive),
        Box::new(ArkRadix2),

        Box::new(ArkRadix2Rec),
        Box::new(LambdaBowers),
        Box::new(WinterfellSplitRadix),
        Box::new(WinterfellFourStep),
        Box::new(Plonky3Radix2DitParallel),
        Box::new(Plonky3Radix2LayerSplit),
        Box::new(LambdaRadix4),
        Box::new(TfheStockhamRadix8),
        Box::new(Fft3w),
    ]
}

//* Note: LambdaRadix4 only included when log_N is even
#[allow(non_snake_case)]
pub fn bench_encoders<F: FftField + Send + Sync>(N: usize) -> Vec<Box<dyn NttEncoder<F>>> {
    let log_n = N.trailing_zeros();
    let mut v: Vec<Box<dyn NttEncoder<F>>> = vec![
        Box::new(ArkRadix2),
        Box::new(ArkRadix2Rec),
        Box::new(LambdaBowers),
        Box::new(WinterfellSplitRadix),
        Box::new(WinterfellFourStep),
        Box::new(Plonky3Radix2DitParallel),
        Box::new(Plonky3Radix2LayerSplit),
        Box::new(Fft3w),
    ];
    if log_n % 2 == 0 {
        v.push(Box::new(LambdaRadix4));
    }
    if log_n % 3 == 0 {
        v.push(Box::new(TfheStockhamRadix8));
    }
    v
}
