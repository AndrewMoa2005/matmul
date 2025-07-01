subroutine matrix_multiply_float(n, a, b, c)
   implicit none
   integer :: n
   real*4, intent(in) :: a(n, n), b(n, n)
   real*4, intent(out) :: c(n, n)
   c = matmul(a, b)
end subroutine matrix_multiply_float

subroutine matrix_multiply_double(n, a, b, c)
   implicit none
   integer :: n
   real*8, intent(in) :: a(n, n), b(n, n)
   real*8, intent(out) :: c(n, n)
   c = matmul(a, b)
end subroutine matrix_multiply_double
