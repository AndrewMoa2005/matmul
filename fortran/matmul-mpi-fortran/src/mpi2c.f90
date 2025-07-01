subroutine matrix_multiply_float(n, rank, size, local_A, B, local_C) bind(C, name="matrix_multiply_float")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), value, intent(in) :: n, rank, size
    real(c_float), intent(in) :: local_A(*)
    real(c_float), intent(in) :: B(*)
    real(c_float), intent(out) :: local_C(*)
    real(c_float), allocatable :: local_A_2d(:, :)
    real(c_float), allocatable :: B_2d(:, :)
    real(c_float), allocatable :: local_C_2d(:, :)
    integer(c_int) :: rows_per_process
    integer(c_int) :: remainder
    integer(c_int) :: i, j

    remainder = mod(n, size)
    rows_per_process = n / size
    if (rank < remainder) then
        rows_per_process = rows_per_process + 1
    end if

    allocate(local_A_2d(rows_per_process, n))
    allocate(B_2d(n, n))
    allocate(local_C_2d(rows_per_process, n))

    ! 将一维数组转换为二维数组
    do i = 1, rows_per_process
        do j = 1, n
            local_A_2d(i, j) = local_A((i - 1) * n + j)
        end do
    end do

    do i = 1, n
        do j = 1, n
            B_2d(i, j) = B((i - 1) * n + j)
        end do
    end do

    ! 使用 matmul 进行矩阵乘法计算
    local_C_2d = matmul(local_A_2d, B_2d)

    ! 将二维数组转换为一维数组
    do i = 1, rows_per_process
        do j = 1, n
            local_C((i - 1) * n + j) = local_C_2d(i, j)
        end do
    end do

    deallocate(local_A_2d, B_2d, local_C_2d)
end subroutine matrix_multiply_float

subroutine matrix_multiply_double(n, rank, size, local_A, B, local_C) bind(C, name="matrix_multiply_double")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int), value, intent(in) :: n, rank, size
    real(c_double), intent(in) :: local_A(*)
    real(c_double), intent(in) :: B(*)
    real(c_double), intent(out) :: local_C(*)
    real(c_double), allocatable :: local_A_2d(:, :)
    real(c_double), allocatable :: B_2d(:, :)
    real(c_double), allocatable :: local_C_2d(:, :)
    integer(c_int) :: rows_per_process
    integer(c_int) :: remainder
    integer(c_int) :: i, j

    remainder = mod(n, size)
    rows_per_process = n / size
    if (rank < remainder) then
        rows_per_process = rows_per_process + 1
    end if

    allocate(local_A_2d(rows_per_process, n))
    allocate(B_2d(n, n))
    allocate(local_C_2d(rows_per_process, n))

    ! 将一维数组转换为二维数组
    do i = 1, rows_per_process
        do j = 1, n
            local_A_2d(i, j) = local_A((i - 1) * n + j)
        end do
    end do

    do i = 1, n
        do j = 1, n
            B_2d(i, j) = B((i - 1) * n + j)
        end do
    end do

    ! 使用 matmul 进行矩阵乘法计算
    local_C_2d = matmul(local_A_2d, B_2d)

    ! 将二维数组转换为一维数组
    do i = 1, rows_per_process
        do j = 1, n
            local_C((i - 1) * n + j) = local_C_2d(i, j)
        end do
    end do

    deallocate(local_A_2d, B_2d, local_C_2d)
end subroutine matrix_multiply_double