use bench::{BenchParams, NttDomain, bench_encoders, gen_input_seeded};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ntt::DefaultField;

fn full_ntt_iter_n(c: &mut Criterion) {
    for params in BenchParams::n_iter(&[24, 27]) {
        let n = params.N;
        let log_n = n.trailing_zeros();
        let domain = NttDomain::<DefaultField>::new(n);
        let input = gen_input_seeded::<DefaultField>(&params, 42);

        let mut group = c.benchmark_group(format!("full_ntt_iter_n/N=2^{log_n}"));
        group.sample_size(10);

        for encoder in bench_encoders::<DefaultField>(n) {
            group.bench_function(encoder.name(), |b| {
                b.iter_batched(|| input.clone(), |mut buf| encoder.ntt_full(&mut buf, &domain), BatchSize::LargeInput)
            });
        }
        group.finish();
    }
}

criterion_group!(benches, full_ntt_iter_n);
criterion_main!(benches);
