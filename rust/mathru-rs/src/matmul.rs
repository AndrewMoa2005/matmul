use mathru::algebra::linear::matrix::General;

pub fn matrix_multiply_float(n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let a_mat = General::new(n, n, a.to_vec());
    let b_mat = General::new(n, n, b.to_vec());
    let c_mat = &a_mat * &b_mat;
    c.copy_from_slice(&c_mat.convert_to_vec());
}

pub fn matrix_multiply_double(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let a_mat = General::new(n, n, a.to_vec());
    let b_mat = General::new(n, n, b.to_vec());
    let c_mat = a_mat * b_mat;
    c.copy_from_slice(&c_mat.convert_to_vec());
}
