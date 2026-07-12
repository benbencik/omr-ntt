use std::time::Duration;

use ark_ff::FftField;
use criterion::{BenchmarkGroup, Throughput, measurement::WallTime};
use ntt::encoders::{
    ArkRadix2, ArkRadix2Rec, Fft3w, LambdaBowers, LambdaRadix4, Plonky3Radix2DitParallel,
    Plonky3Radix2LayerSplit, TfheStockhamRadix8, WinterfellFourStep, WinterfellSplitRadix,
};
use rand::SeedableRng;

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

    pub fn n_iter(log_ns: &[u32]) -> Vec<Self> {
        const S: usize = 1000; // fix to s 1000 (arbitrary) 
        log_ns
            .iter()
            .map(|&log_n| Self::new(1 << log_n, S))
            .collect()
    }

    pub fn s_iter(s_values: &[usize]) -> Vec<Self> {
        const LOG_N: u32 = 22; // fix to logn 22 (arbitrary)
        s_values.iter().map(|&s| Self::new(1 << LOG_N, s)).collect()
    }
}

pub fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>, n: usize) {
    let secs = if n >= 1 << 26 { 300 } else { 60 };
    group
        .measurement_time(Duration::from_secs(secs))
        .throughput(Throughput::Elements(n as u64));
}

pub fn gen_input_seeded<F: FftField>(params: &BenchParams, seed: u64) -> Input<F> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let v = (0..params.N).map(|_| F::rand(&mut rng)).collect();
    v
}

pub fn all_implemented_encoders<F: FftField + Send + Sync>() -> Vec<Box<dyn NttEncoder<F>>> {
    vec![
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
