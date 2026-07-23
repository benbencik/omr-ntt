//! Standalone profiling harness (fixed logN=27, s=50). Good under perf/flamegraph.
//! Run: `direnv exec . cargo run --release --example profile_ntt`
use ark_ff::FftField;
use ntt::encoder::NttDomain;
use ntt::encoders::TransformDecomposition;
use ntt::{DefaultField, NttEncoder};
use rand::SeedableRng;
use std::hint::black_box;
use std::time::Duration;

const LOG_N: u32 = 27;
const S: usize = 50;
const ITERS: usize = 5;

fn gen_input<F: FftField>(n: usize, seed: u64) -> Vec<F> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    (0..n).map(|_| F::rand(&mut rng)).collect()
}

fn main() {
    let n = 1usize << LOG_N;
    let domain = NttDomain::<DefaultField>::new(n);
    let encoder = TransformDecomposition::new(S);

    // warm up
    let mut warmup_input_1 = gen_input::<DefaultField>(n, 40);
    let mut warmup_input_2 = gen_input::<DefaultField>(n, 41);
    encoder.ntt(black_box(&mut warmup_input_1), black_box(&domain));
    encoder.ntt(black_box(&mut warmup_input_2), black_box(&domain));

    // clone all inputs, before the main run
    let input = gen_input::<DefaultField>(n, 42);
    let mut buff: Vec<_> = (0..ITERS).map(|_| input.clone()).collect();

    // pause to see gap in profiling
    std::thread::sleep(Duration::from_secs(2));

    for b in &mut buff {
        encoder.ntt(black_box(b), black_box(&domain));
    }
    let _ = black_box(&buff);
}
