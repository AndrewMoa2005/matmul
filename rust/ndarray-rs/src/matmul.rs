use ndarray::{Array2, ArrayView2};

pub fn matrix_multiply_float(n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    // 将一维数组转为二维ndarray视图
    let a = ArrayView2::from_shape((n, n), a).unwrap();
    let b = ArrayView2::from_shape((n, n), b).unwrap();
    let mut c_mat = Array2::<f32>::zeros((n, n));
    // 矩阵相乘
    c_mat.assign(&a.dot(&b));
    // 将结果转回一维数组
    c.copy_from_slice(c_mat.as_slice().unwrap());
}

pub fn matrix_multiply_double(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let a = ArrayView2::from_shape((n, n), a).unwrap();
    let b = ArrayView2::from_shape((n, n), b).unwrap();
    let mut c_mat = Array2::<f64>::zeros((n, n));
    c_mat.assign(&a.dot(&b));
    c.copy_from_slice(c_mat.as_slice().unwrap());
}
