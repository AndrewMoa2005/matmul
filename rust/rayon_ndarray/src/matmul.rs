use ndarray::{ArrayView2, s};
use rayon::prelude::*;

pub fn matrix_multiply_float(n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let b_view = ArrayView2::from_shape((n, n), b).unwrap();
    let a_view = ArrayView2::from_shape((n, n), a).unwrap();

    // 动态计算分块大小
    let num_blocks = rayon::current_num_threads();
    let block_size = (n + num_blocks - 1) / num_blocks; // 向上取整分块

    c.par_chunks_mut(n * block_size)
        .enumerate()
        .for_each(|(block_idx, c_block)| {
            let start_row = block_idx * block_size;
            let end_row = (start_row + block_size).min(n);
            let rows_to_process = end_row - start_row;

            let a_block = a_view.slice(s![start_row..end_row, ..]);
            let c_mat = a_block.dot(&b_view);
            c_block[..rows_to_process * n].copy_from_slice(c_mat.as_slice().unwrap());
        });
}

pub fn matrix_multiply_double(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let b_view = ArrayView2::from_shape((n, n), b).unwrap();
    let a_view = ArrayView2::from_shape((n, n), a).unwrap();

    // 动态计算分块大小
    let num_blocks = rayon::current_num_threads();
    let block_size = (n + num_blocks - 1) / num_blocks; // 向上取整分块

    c.par_chunks_mut(n * block_size)
        .enumerate()
        .for_each(|(block_idx, c_block)| {
            let start_row = block_idx * block_size;
            let end_row = (start_row + block_size).min(n);
            let rows_to_process = end_row - start_row;

            let a_block = a_view.slice(s![start_row..end_row, ..]);
            let c_mat = a_block.dot(&b_view);
            c_block[..rows_to_process * n].copy_from_slice(c_mat.as_slice().unwrap());
        });
}
