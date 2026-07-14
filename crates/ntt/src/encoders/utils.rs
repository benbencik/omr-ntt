use rayon::prelude::*;

pub(super) fn derange<T: Send>(xi: &mut [T], log_len: u32) {
    let n = xi.len();
    // cast to usize to bypass compiler checks, since pointers do not implement `Sync`
    let ptr = xi.as_mut_ptr() as usize;
    // skip first (0...0) and last (1...1) elements
    (1..n as u64 - 1).into_par_iter().for_each(|idx| {
        let rev_idx = bitrev(idx, log_len) as usize;
        // swap is applied only once per pair
        if idx < rev_idx as u64 {            
            // This is safe because bitrev is a bijection
            unsafe {
                let p = ptr as *mut T;
                std::ptr::swap(p.add(idx as usize), p.add(rev_idx));
            }
        }
    });
}

#[inline]
// reverse first log_len bits of a
pub(super) fn bitrev(a: u64, log_len: u32) -> u64 {
    a.reverse_bits() >> (64 - log_len)
}
