subroutine matrix_multiply_float(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*4, intent(in) :: a(n, full_dim), b(full_dim, full_dim)
   real*4, intent(out) :: c(n, full_dim)
   ! 使用 matmul 函数进行矩阵乘法
   c = matmul(a, b)
end subroutine matrix_multiply_float

subroutine matrix_multiply_double(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*8, intent(in) :: a(n, full_dim), b(full_dim, full_dim)
   real*8, intent(out) :: c(n, full_dim)
   ! 使用 matmul 函数进行矩阵乘法
   c = matmul(a, b)
end subroutine matrix_multiply_double