use bench::{BenchParams, NttDomain, configure_group, gen_input_seeded};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ntt::{DefaultField, encoders};

fn partial_ntt_iter_s(c: &mut Criterion) {
    for params in BenchParams::s_iter(&[22, 24, 27], &[50, 100, 1000]) {
        let n = params.N;
        let s = params.s;
        let log_n = n.trailing_zeros();
        let domain = NttDomain::<DefaultField>::new(n);
        let input = gen_input_seeded::<DefaultField>(&params, 42);

        let mut group = c.benchmark_group(format!("partial_ntt_iter_s/logN={log_n}_s={s}"));
        configure_group(&mut group, n);
        group.sample_size(50);

        for encoder in encoders::all_partial::<DefaultField>(s) {
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

criterion_group!(benches, partial_ntt_iter_s);
criterion_main!(benches);
