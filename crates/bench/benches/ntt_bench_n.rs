use bench::{BenchParams, NttDomain, configure_group, gen_input_seeded};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ntt::{DefaultField, encoders};

fn full_ntt_iter_n(c: &mut Criterion) {
    for params in BenchParams::n_iter(&[22, 24, 27]) {
        let n = params.N;
        let log_n = n.trailing_zeros();
        let domain = NttDomain::<DefaultField>::new(n);
        let input = gen_input_seeded::<DefaultField>(&params, 42);

        let mut group = c.benchmark_group(format!("full_ntt_iter_n/N=2^{log_n}"));
        configure_group(&mut group, n);

        for encoder in encoders::all::<DefaultField>(log_n) {
            group.bench_function(encoder.name(), |b| {
                b.iter_batched(
                    || input.clone(),
                    |mut buf| encoder.ntt(&mut buf, &domain),
                    BatchSize::LargeInput,
                )
            });
        }
        group.finish();
    }
}

criterion_group!(benches, full_ntt_iter_n);
criterion_main!(benches);
