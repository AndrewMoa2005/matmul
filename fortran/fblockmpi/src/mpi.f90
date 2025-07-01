subroutine matrix_multiply_float(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*4, intent(in) :: a(n * full_dim), b(full_dim * full_dim)
   real*4, intent(out) :: c(n * full_dim)
   integer :: i, j, k, ii, jj, kk, block_size
   
   ! 直接使用一维数组索引计算，避免内存拷贝
   block_size = 8  ! 可根据CPU缓存大小调整
   c = 0.0
   
   do ii = 1, n, block_size
      do jj = 1, full_dim, block_size
         do kk = 1, full_dim, block_size
            do i = ii, min(ii+block_size-1, n)
               do j = jj, min(jj+block_size-1, full_dim)
                  do k = kk, min(kk+block_size-1, full_dim)
                     c((i-1)*full_dim + j) = c((i-1)*full_dim + j) + a((i-1)*full_dim + k) * b((k-1)*full_dim + j)
                  end do
               end do
            end do
         end do
      end do
   end do
end subroutine matrix_multiply_float

subroutine matrix_multiply_double(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*8, intent(in) :: a(n * full_dim), b(full_dim * full_dim)
   real*8, intent(out) :: c(n * full_dim)
   integer :: i, j, k, ii, jj, kk, block_size
   
   ! 直接使用一维数组索引计算，避免内存拷贝
   block_size = 8  ! 可根据CPU缓存大小调整
   c = 0.0d0
   
   do ii = 1, n, block_size
      do jj = 1, full_dim, block_size
         do kk = 1, full_dim, block_size
            do i = ii, min(ii+block_size-1, n)
               do j = jj, min(jj+block_size-1, full_dim)
                  do k = kk, min(kk+block_size-1, full_dim)
                     c((i-1)*full_dim + j) = c((i-1)*full_dim + j) + a((i-1)*full_dim + k) * b((k-1)*full_dim + j)
                  end do
               end do
            end do
         end do
      end do
   end do
end subroutine matrix_multiply_double