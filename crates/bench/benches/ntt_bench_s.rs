use bench::{BenchParams, NttDomain, bench_encoders, configure_group, gen_input_seeded};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ntt::DefaultField;

fn full_ntt_iter_s(c: &mut Criterion) {
    for params in BenchParams::s_iter(&[100, 500, 1000, 2000, 5000]) {
        let n = params.N;
        let s = params.s;
        let log_n = n.trailing_zeros();
        let domain = NttDomain::<DefaultField>::new(n);
        let input = gen_input_seeded::<DefaultField>(&params, 42);

        let mut group = c.benchmark_group(format!("full_ntt_iter_s/N=2^{log_n}_s={s}"));
        configure_group(&mut group, n);

        for encoder in bench_encoders::<DefaultField>(&domain) {
            group.bench_function(encoder.name(), |b| {
                b.iter_batched(
                    || input.clone(),
                    |mut buf| encoder.ntt_full(&mut buf, &domain),
                    BatchSize::LargeInput,
                )
            });
        }
        group.finish();
    }
}

criterion_group!(benches, full_ntt_iter_s);
criterion_main!(benches);
