// All NTT encoder implementations

mod ark_radix2;
mod ark_transforms_radix2_rec;
mod fft3w;
mod lambdaworks_bowers;
mod lambdaworks_radix4;
mod dft;
mod plonky3_radix2_dit_parallel;
mod plonky3_radix2_layer_split;
mod tfhe_stockham_radix8;
mod transform_decomposition;
mod transpose_out_of_place;
mod utils;
mod winterfell_four_step;

pub use ark_radix2::ArkRadix2;
pub use ark_transforms_radix2_rec::ArkRadix2Rec;
pub use fft3w::Fft3w;
pub use lambdaworks_bowers::LambdaBowers;
pub use lambdaworks_radix4::LambdaRadix4;
pub use dft::{Dft, DftPartial};
pub use plonky3_radix2_dit_parallel::Plonky3Radix2DitParallel;
pub use plonky3_radix2_layer_split::Plonky3Radix2LayerSplit;
pub use tfhe_stockham_radix8::TfheStockhamRadix8;
pub use transform_decomposition::TransformDecomposition;
pub use winterfell_four_step::WinterfellFourStep;

use ark_ff::FftField;

use crate::encoder::NttEncoder;

// LambdaRadix4 only included when log_N is even
// TfheStockhamRadix8 only included when log_N is divisible by 3
// Dft is excluded, too slow to bench for large N
pub fn all<F: FftField + Send + Sync>(log_n: u32) -> Vec<Box<dyn NttEncoder<F>>> {
    let mut v: Vec<Box<dyn NttEncoder<F>>> = vec![
        Box::new(ArkRadix2),
        Box::new(ArkRadix2Rec),
        Box::new(LambdaBowers),
        Box::new(WinterfellFourStep),
        Box::new(Plonky3Radix2DitParallel),
        Box::new(Plonky3Radix2LayerSplit),
        Box::new(Fft3w),
    ];
    if log_n % 2 == 0 {
        v.push(Box::new(LambdaRadix4));
    }
    if log_n % 3 == 0 {
        v.push(Box::new(TfheStockhamRadix8));
    }
    v
}

pub fn all_partial<F: FftField + Send + Sync>(s: usize) -> Vec<Box<dyn NttEncoder<F>>> {
    vec![
        Box::new(DftPartial::new(s)),
        Box::new(TransformDecomposition::new(s)),
    ]
}
