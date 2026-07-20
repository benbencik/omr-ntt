use bench::{BenchParams, NttDomain, configure_group, gen_input_seeded};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ntt::encoders::TransformDecomposition;
use ntt::{DefaultField, NttEncoder};

fn partial_ntt_sweep_s(c: &mut Criterion) {
    let log_n: u32 = 27;
    let s_values = [50, 500, 5000];

    let n = 1 << log_n;
    let domain = NttDomain::<DefaultField>::new(n);

    let mut group = c.benchmark_group(format!("partial_ntt_sweep_s/logN={log_n}"));
    configure_group(&mut group, n);

    for &s in &s_values {
        let params = BenchParams::new(n, s);
        let input = gen_input_seeded::<DefaultField>(&params, 42);
        let encoder = TransformDecomposition::new(s);

        group.bench_function(format!("s={s}"), |b| {
            b.iter_batched(
                || input.clone(),
                |mut buf| encoder.ntt(&mut buf, &domain),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, partial_ntt_sweep_s);
criterion_main!(benches);
