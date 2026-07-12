use bench::{BenchParams, configure_group};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_goldilocks::Goldilocks as P3Gold;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use winter_math::fft::{evaluate_poly, get_twiddles};
use winter_math::fields::f64::BaseElement as WinterGold;

// Common field: Goldilocks 
const P: u64 = 18446744069414584321; 

fn bench_extern_ntt(c: &mut Criterion) {
    for params in BenchParams::n_iter(&[22, 24, 27]) {
        let n = params.N;
        let log_n = n.trailing_zeros();
        let mut rng = SmallRng::seed_from_u64(42);

        let input_tfhe: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>() % P).collect();
        let input_p3: Vec<P3Gold> = input_tfhe.iter().map(|&x| P3Gold::new(x)).collect();
        let input_winterfell: Vec<WinterGold> = input_tfhe.iter().map(|&x| WinterGold::new(x)).collect();

        let mut group = c.benchmark_group(format!("extern_ntt/N=2^{log_n}"));
        configure_group(&mut group, n);

        let plan = tfhe_ntt::prime64::Plan::try_new(n, P)
            .expect("tfhe-ntt: Goldilocks NTT plan failed");
        group.bench_function("TFHE", |b| {
            b.iter_batched(
                || input_tfhe.clone(),
                |mut v| plan.fwd(&mut v),
                BatchSize::LargeInput,
            )
        });

        let p3_dft = Radix2DitParallel::<P3Gold>::default();
        group.bench_function("Plonky3", |b| {
            b.iter_batched(
                || input_p3.clone(),
                |v| p3_dft.dft(v),
                BatchSize::LargeInput,
            )
        });
        
        // TODO: evaluates over 2N-th roots (odd powers) output is differnt, investigate more
        let twiddles = get_twiddles::<WinterGold>(n);
        group.bench_function("Winterfell", |b| {
            b.iter_batched(
                || input_winterfell.clone(),
                |mut v| evaluate_poly(&mut v, &twiddles),
                BatchSize::LargeInput,
            )
        });

        group.finish();
    }
}

criterion_group!(benches, bench_extern_ntt);
criterion_main!(benches);
