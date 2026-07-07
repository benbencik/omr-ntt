use ark_std::test_rng;
use bench::{BenchParams, NttDomain, bench_encoders, gen_sparse_input};
use criterion::{Criterion, criterion_group, criterion_main};
use ntt::DefaultField;

fn full_ntt_iter_n(c: &mut Criterion) {
    for params in BenchParams::n_sweep(&[27]) {
        let n = params.N;
        let log_n = n.trailing_zeros();
        let domain = NttDomain::<DefaultField>::new(n);
        let input = gen_sparse_input::<DefaultField>(&params, &mut test_rng());

        let mut group = c.benchmark_group(format!("full_ntt_iter_n/N=2e{log_n}"));
        group.sample_size(10);

        for encoder in bench_encoders::<DefaultField>(n) {
            group.bench_function(encoder.name(), |b| {
                b.iter(|| encoder.ntt_full(&input, &domain))
            });
        }
        group.finish();
    }
}

criterion_group!(benches, full_ntt_iter_n);
criterion_main!(benches);
