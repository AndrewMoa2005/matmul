subroutine matrix_multiply_float(n, a, b, c)
   implicit none
   integer, intent(in) :: n
   real*4, intent(in) :: a(n, n), b(n, n)
   real*4, intent(out) :: c(n, n)
   integer :: i, j, k, bi, bj, bk, block_size
   real*4 :: temp

   ! 定义块大小，可根据实际情况调整
   block_size = 64

   c = 0.0
   !$omp parallel do private(bi, bj, bk, i, j, k, temp) shared(a, b, c, n, block_size)
   do bi = 1, n, block_size
      do bj = 1, n, block_size
         do bk = 1, n, block_size
            do i = bi, min(bi + block_size - 1, n)
               do j = bj, min(bj + block_size - 1, n)
                  temp = c(i, j)
                  do k = bk, min(bk + block_size - 1, n)
                     temp = temp + a(i, k) * b(k, j)
                  end do
                  c(i, j) = temp
               end do
            end do
         end do
      end do
   end do
   !$omp end parallel do
end subroutine matrix_multiply_float

subroutine matrix_multiply_double(n, a, b, c)
   implicit none
   integer, intent(in) :: n
   real*8, intent(in) :: a(n, n), b(n, n)
   real*8, intent(out) :: c(n, n)
   integer :: i, j, k, bi, bj, bk, block_size
   real*8 :: temp

   block_size = 64

   c = 0.0d0
   !$omp parallel do private(bi, bj, bk, i, j, k, temp) shared(a, b, c, n, block_size)
   do bi = 1, n, block_size
      do bj = 1, n, block_size
         do bk = 1, n, block_size
            do i = bi, min(bi + block_size - 1, n)
               do j = bj, min(bj + block_size - 1, n)
                  temp = c(i, j)
                  do k = bk, min(bk + block_size - 1, n)
                     temp = temp + a(i, k) * b(k, j)
                  end do
                  c(i, j) = temp
               end do
            end do
         end do
      end do
   end do
   !$omp end parallel do
end subroutine matrix_multiply_double