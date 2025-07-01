subroutine matrix_multiply_float(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*4, intent(in) :: a(n * full_dim), b(full_dim * full_dim)
   real*4, intent(out) :: c(n * full_dim)
   real*4 :: a_2d(n, full_dim), b_2d(full_dim, full_dim), c_2d(n, full_dim)
   integer :: i, j, k
   
   ! 使用reshape函数转换一维数组为二维数组
   a_2d = transpose(reshape(a, [full_dim, n]))
   b_2d = transpose(reshape(b, [full_dim, full_dim]))
   
   ! 矩阵乘法
   c_2d = matmul(a_2d, b_2d)
   
   ! 使用reshape函数转换二维数组为一维数组
   c = reshape(transpose(c_2d), [n * full_dim])
end subroutine matrix_multiply_float

subroutine matrix_multiply_double(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*8, intent(in) :: a(n * full_dim), b(full_dim * full_dim)
   real*8, intent(out) :: c(n * full_dim)
   real*8 :: a_2d(n, full_dim), b_2d(full_dim, full_dim), c_2d(n, full_dim)
   integer :: i, j, k
   
   ! 使用reshape函数转换一维数组为二维数组
   a_2d = transpose(reshape(a, [full_dim, n]))
   b_2d = transpose(reshape(b, [full_dim, full_dim]))
   
   ! 矩阵乘法
   c_2d = matmul(a_2d, b_2d)
   
   ! 使用reshape函数转换二维数组为一维数组
   c = reshape(transpose(c_2d), [n * full_dim])
end subroutine matrix_multiply_double