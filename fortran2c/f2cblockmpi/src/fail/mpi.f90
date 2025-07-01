subroutine matrix_multiply_float(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*4, intent(in) :: a(n, full_dim), b(full_dim, full_dim)
   real*4, intent(out) :: c(n, full_dim)
   integer :: i, j, k, ii, jj, kk
   real*4 :: temp
   integer, parameter :: block_size = 16 ! 定义固定分块大小为
   c = 0.0
   ! 分块遍历
   do ii = 1, n, block_size
      do jj = 1, full_dim, block_size
         do kk = 1, full_dim, block_size
            ! 块内计算
            do i = ii, min(ii+block_size-1, n)
               do j = jj, min(jj+block_size-1, full_dim)
                  temp = c(i, j)
                  do k = kk, min(kk+block_size-1, full_dim)
                     temp = temp + a(i, k) * b(k, j)
                  end do
                  c(i, j) = temp
               end do
            end do
         end do
      end do
   end do
end subroutine matrix_multiply_float

subroutine matrix_multiply_double(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*8, intent(in) :: a(n, full_dim), b(full_dim, full_dim)
   real*8, intent(out) :: c(n, full_dim)
   integer :: i, j, k, ii, jj, kk
   real*8 :: temp
   integer, parameter :: block_size = 16 ! 定义固定分块大小为
   c = 0.0d0
   ! 分块遍历
   do ii = 1, n, block_size
      do jj = 1, full_dim, block_size
         do kk = 1, full_dim, block_size
            ! 块内计算
            do i = ii, min(ii+block_size-1, n)
               do j = jj, min(jj+block_size-1, full_dim)
                  temp = c(i, j)
                  do k = kk, min(kk+block_size-1, full_dim)
                     temp = temp + a(i, k) * b(k, j)
                  end do
                  c(i, j) = temp
               end do
            end do
         end do
      end do
   end do
end subroutine matrix_multiply_double