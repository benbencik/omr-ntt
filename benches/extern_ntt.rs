mod common;
use common::{BenchParams, NttDomain, NttEncoder, configure_group, gen_input_seeded};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ntt::Goldilocks;
use ntt::encoders::{TransformDecomposition, WinterfellFourStep};
use oxifft::{Complex as OxiComplex, Direction, Flags, Plan as OxiPlan};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_goldilocks::Goldilocks as P3Gold;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rustfft::{FftPlanner, num_complex::Complex as RustComplex};
use winter_math::fft::{evaluate_poly, get_twiddles};
use winter_math::fields::f64::BaseElement as WinterGold;

// Goldilocks prime (2^64 - 2^32 + 1)
const P: u64 = 18446744069414584321;

// s contant for partial ntt
const S_PARTIAL: usize = 50;

fn bench_ntt(c: &mut Criterion) {
    for params in BenchParams::n_iter(&[27]) {
        let n = params.N;
        let log_n = n.trailing_zeros();
        let mut rng = SmallRng::seed_from_u64(42);

        // Goldilocks inputs set for NTTs to use the same field
        let input_u64: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>() % P).collect();
        let input_p3: Vec<P3Gold> = input_u64.iter().map(|&x| P3Gold::new(x)).collect();
        let input_winterfell: Vec<WinterGold> =
            input_u64.iter().map(|&x| WinterGold::new(x)).collect();

        // Float inputs for RustFFT and oxifft (do not support NTT)
        let mut rng_f = SmallRng::seed_from_u64(43);
        let rust_input: Vec<RustComplex<f64>> = (0..n)
            .map(|_| RustComplex {
                re: rng_f.r#gen::<f64>(),
                im: 0.0,
            })
            .collect();
        let oxi_input: Vec<OxiComplex<f64>> = rust_input
            .iter()
            .map(|c| OxiComplex::new(c.re, c.im))
            .collect();

        // Goldilocks input for our local encoders — same field as external libs.
        let field_input = gen_input_seeded::<Goldilocks>(&params, 42);
        let domain = NttDomain::<Goldilocks>::new(n);

        // setup for external libs
        let tfhe_plan = tfhe_ntt::prime64::Plan::try_new(n, P).expect("tfhe-ntt plan failed");
        let p3_dft = Radix2DitParallel::<P3Gold>::default();
        let twiddles = get_twiddles::<WinterGold>(n);
        let mut planner = FftPlanner::<f64>::new();
        let fft_plan = planner.plan_fft_forward(n);
        let oxi_plan = OxiPlan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE)
            .expect("oxifft plan failed");
        let partial_encoder = TransformDecomposition::new(S_PARTIAL);

        let mut group = c.benchmark_group(format!("extern_ntt/logN={log_n}"));
        configure_group(&mut group, n);
        group.sample_size(10);

        // Port of Winterfell NTT Goldilocks u64 with better matrix transpose
        group.bench_function("WinterfellFourStep local port (Goldilocks u64)", |b| {
            b.iter_batched(
                || field_input.clone(),
                |mut buf| WinterfellFourStep.ntt(&mut buf, &domain),
                BatchSize::LargeInput,
            )
        });

        // Full NTT, Goldilocks u64
        group.bench_function("TFHE (Goldilocks u64)", |b| {
            b.iter_batched(
                || input_u64.clone(),
                |mut v| tfhe_plan.fwd(&mut v),
                BatchSize::LargeInput,
            )
        });

        // Full NTT, Goldilocks u64
        group.bench_function("Plonky3 (Goldilocks u64)", |b| {
            b.iter_batched(
                || input_p3.clone(),
                |v| p3_dft.dft(v),
                BatchSize::LargeInput,
            )
        });

        // Full NTT, Goldilocks u64
        group.bench_function("Winterfell (Goldilocks u64)", |b| {
            b.iter_batched(
                || input_winterfell.clone(),
                |mut v| evaluate_poly(&mut v, &twiddles),
                BatchSize::LargeInput,
            )
        });

        // Full NTT, Complex f64
        group.bench_function("RustFFT (Complex f64)", |b| {
            b.iter_batched(
                || rust_input.clone(),
                |mut buf| fft_plan.process(&mut buf),
                BatchSize::LargeInput,
            )
        });

        // Pruned feature on, Complex f64
        group.bench_function("oxifft (Complexf64)", |b| {
            b.iter_batched(
                || (oxi_input.clone(), vec![OxiComplex::<f64>::zero(); n]),
                |(inp, mut out)| oxi_plan.execute(&inp, &mut out),
                BatchSize::LargeInput,
            )
        });

        // Local transpose decomposition first 2*s outputs, Goldilocks u64
        group.bench_function(
            "TransformDecomposition s=50 partial (Goldilocks u64)",
            |b| {
                b.iter_batched(
                    || field_input.clone(),
                    |mut buf| partial_encoder.ntt(&mut buf, &domain),
                    BatchSize::LargeInput,
                )
            },
        );

        group.finish();
    }
}

criterion_group!(benches, bench_ntt);
criterion_main!(benches);
