use ark_std::test_rng;
use bench::{BenchParams, NttDomain, bench_encoders, gen_sparse_input};
use criterion::{Criterion, criterion_group, criterion_main};
use ntt::DefaultField;

fn full_ntt_iter_s(c: &mut Criterion) {
    for params in BenchParams::s_iter(&[100, 500, 1000, 2000, 5000]) {
        let n = params.N;
        let s = params.s;
        let log_n = n.trailing_zeros();
        let domain = NttDomain::<DefaultField>::new(n);
        let input = gen_sparse_input::<DefaultField>(&params, &mut test_rng());

        let mut group = c.benchmark_group(format!("full_ntt_iter_s/N=2^{log_n}_s={s}"));
        group.sample_size(10);

        for encoder in bench_encoders::<DefaultField>(n) {
            group.bench_function(encoder.name(), |b| {
                b.iter(|| encoder.ntt_full(&input, &domain))
            });
        }
        group.finish();
    }
}

criterion_group!(benches, full_ntt_iter_s);
criterion_main!(benches);
