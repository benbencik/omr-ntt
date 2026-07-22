use ark_ff::FftField;
use rayon::prelude::*;

use crate::encoder::{NttDomain, NttEncoder};

// Computes W[j] by brute-force used for testing other implementations
pub struct Dft;

impl<F: FftField> NttEncoder<F> for Dft {
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>) {
        let n = domain.N;
        let input: Vec<F> = buf.to_vec();
        for j in 0..n {
            let step = domain.omega.pow([j as u64]);
            let mut t = F::one();
            buf[j] = F::zero();
            for &v in &input {
                buf[j] += v * t;
                t *= step;
            }
        }
    }

    fn name(&self) -> &str {
        "Dft"
    }
}

// Computes only the first 2*s outputs W[0..2s].
pub struct DftPartial {
    pub s: usize,
}

impl DftPartial {
    pub fn new(s: usize) -> Self {
        Self { s }
    }
}

impl<F: FftField + Send + Sync> NttEncoder<F> for DftPartial {
    fn ntt(&self, buf: &mut [F], domain: &NttDomain<F>) {
        let n = domain.N;
        let out_len = (2 * self.s).min(n);
        if out_len == 0 {
            return;
        }
        let omega = domain.omega;

        // parallelize over the input N rather than the 2s outputs
        // there are too few outputs to keep all cores busy
        // each chunk accumulates partial sums, then reduce

        // TODO: the chunk should be adaptive based on the out length and maybe the number of cores idk
        // this is faster than transform decompose on really small s could be probably improved
        const CHUNK: usize = 1 << 14;
        let acc = buf
            .par_chunks(CHUNK)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                // chunks cover part of each 2s outputs
                // for each outout j, sum results over all chunks
                let start = (chunk_idx * CHUNK) as u64;
                let mut local = vec![F::zero(); out_len];

                for (j, slot) in local.iter_mut().enumerate() {
                    let step = omega.pow([j as u64]);
                    let mut t = omega.pow([start * j as u64]); // omega^(i*j) at i = start
                    let mut sum = F::zero();
                    for &v in chunk {
                        sum += v * t; // add x[i] * omega^(i*j)
                        t *= step; // advance omega^(i*j) -> omega^((i+1)*j)
                    }
                    *slot = sum;
                }
                local
            })
            // add results from chunks
            .reduce(
                || vec![F::zero(); out_len],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );

        buf[..out_len].copy_from_slice(&acc);
    }

    fn name(&self) -> &str {
        "DftPartial"
    }
}
