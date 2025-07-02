use rayon::prelude::*;

// 根据数据类型调整分块大小
const BLOCK_SIZE: usize = 8;

pub fn matrix_multiply_float(n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_iter_mut().enumerate().for_each(|(idx, c_ij)| {
        let i = idx / n;
        let j = idx % n;
        *c_ij = 0.0;
        for k_block in (0..n).step_by(BLOCK_SIZE) {
            let end = (k_block + BLOCK_SIZE).min(n);
            for k in k_block..end {
                *c_ij += a[i * n + k] * b[k * n + j];
            }
        }
    });
}

pub fn matrix_multiply_double(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    c.par_iter_mut().enumerate().for_each(|(idx, c_ij)| {
        let i = idx / n;
        let j = idx % n;
        *c_ij = 0.0;
        for k_block in (0..n).step_by(BLOCK_SIZE) {
            let end = (k_block + BLOCK_SIZE).min(n);
            for k in k_block..end {
                *c_ij += a[i * n + k] * b[k * n + j];
            }
        }
    });
}
