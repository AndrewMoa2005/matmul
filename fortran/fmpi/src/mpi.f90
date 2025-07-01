subroutine matrix_multiply_float(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*4, intent(in) :: a(n * full_dim), b(full_dim * full_dim)
   real*4, intent(out) :: c(n * full_dim)
   real*4 :: a_2d(n, full_dim), b_2d(full_dim, full_dim), c_2d(n, full_dim)
   integer :: i, j, k
   
   ! 将一维数组转换为二维数组
   do i = 1, n
      do j = 1, full_dim
         a_2d(i, j) = a((i-1)*full_dim + j)
      end do
   end do
   
   do i = 1, full_dim
      do j = 1, full_dim
         b_2d(i, j) = b((i-1)*full_dim + j)
      end do
   end do
   
   ! 矩阵乘法
   c_2d = 0.0
   do i = 1, n
      do j = 1, full_dim
         do k = 1, full_dim
            c_2d(i, j) = c_2d(i, j) + a_2d(i, k) * b_2d(k, j)
         end do
      end do
   end do
   
   ! 将结果转换回一维数组
   do i = 1, n
      do j = 1, full_dim
         c((i-1)*full_dim + j) = c_2d(i, j)
      end do
   end do
end subroutine matrix_multiply_float

subroutine matrix_multiply_double(n, a, b, c, full_dim)
   implicit none
   integer, intent(in) :: n, full_dim
   real*8, intent(in) :: a(n * full_dim), b(full_dim * full_dim)
   real*8, intent(out) :: c(n * full_dim)
   real*8 :: a_2d(n, full_dim), b_2d(full_dim, full_dim), c_2d(n, full_dim)
   integer :: i, j, k
   
   ! 将一维数组转换为二维数组
   do i = 1, n
      do j = 1, full_dim
         a_2d(i, j) = a((i-1)*full_dim + j)
      end do
   end do
   
   do i = 1, full_dim
      do j = 1, full_dim
         b_2d(i, j) = b((i-1)*full_dim + j)
      end do
   end do
   
   ! 矩阵乘法
   c_2d = 0.0d0
   do i = 1, n
      do j = 1, full_dim
         do k = 1, full_dim
            c_2d(i, j) = c_2d(i, j) + a_2d(i, k) * b_2d(k, j)
         end do
      end do
   end do
   
   ! 将结果转换回一维数组
   do i = 1, n
      do j = 1, full_dim
         c((i-1)*full_dim + j) = c_2d(i, j)
      end do
   end do
end subroutine matrix_multiply_double