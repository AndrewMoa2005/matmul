subroutine matrix_multiply_float(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*4, intent(in) :: a(n, full_dim), b(full_dim, full_dim)
   real*4, intent(out) :: c(n, full_dim)
   integer :: i, j, k
   c = 0.0
   do i = 1, n
      do j = 1, full_dim
         do k = 1, full_dim
            c(i, j) = c(i, j) + a(i, k) * b(k, j)
         end do
      end do
   end do
end subroutine matrix_multiply_float

subroutine matrix_multiply_double(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*8, intent(in) :: a(n, full_dim), b(full_dim, full_dim)
   real*8, intent(out) :: c(n, full_dim)
   integer :: i, j, k
   c = 0.0d0
   do i = 1, n
      do j = 1, full_dim
         do k = 1, full_dim
            c(i, j) = c(i, j) + a(i, k) * b(k, j)
         end do
      end do
   end do
end subroutine matrix_multiply_double