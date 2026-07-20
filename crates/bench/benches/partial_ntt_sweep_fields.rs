use bench::{BenchParams, NttDomain, configure_group, gen_input_seeded};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ntt::encoders::TransformDecomposition;
use ntt::fields::{BabyBear, Goldilocks, TeddyBear};
use ntt::NttEncoder;

fn partial_ntt_sweep_fields(c: &mut Criterion) {
    let log_n: u32 = 27;
    let s: usize = 50;
    let n = 1 << log_n;
    let params = BenchParams::new(n, s);

    let mut group = c.benchmark_group(format!("partial_ntt_sweep_fields/logN={log_n}_s={s}"));
    configure_group(&mut group, n);

    // BabyBear
    let domain = NttDomain::<BabyBear>::new(n);
    let input = gen_input_seeded::<BabyBear>(&params, 42);
    let encoder = TransformDecomposition::new(s);
    group.bench_function("BabyBear", |b| {
        b.iter_batched(
            || input.clone(),
            |mut buf| encoder.ntt(&mut buf, &domain),
            BatchSize::LargeInput,
        )
    });

    // TeddyBear
    let domain = NttDomain::<TeddyBear>::new(n);
    let input = gen_input_seeded::<TeddyBear>(&params, 42);
    let encoder = TransformDecomposition::new(s);
    group.bench_function("TeddyBear", |b| {
        b.iter_batched(
            || input.clone(),
            |mut buf| encoder.ntt(&mut buf, &domain),
            BatchSize::LargeInput,
        )
    });

    // Goldilocks
    let domain = NttDomain::<Goldilocks>::new(n);
    let input = gen_input_seeded::<Goldilocks>(&params, 42);
    let encoder = TransformDecomposition::new(s);
    group.bench_function("Goldilocks", |b| {
        b.iter_batched(
            || input.clone(),
            |mut buf| encoder.ntt(&mut buf, &domain),
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(benches, partial_ntt_sweep_fields);
criterion_main!(benches);
