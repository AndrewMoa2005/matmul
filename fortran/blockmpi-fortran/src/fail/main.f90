program main
    use mpi
    implicit none
    external matrix_multiply_float, matrix_multiply_double
    integer :: n = 10         ! 默认矩阵大小指数
    integer :: loop_num = 5   ! 用于求平均值的迭代次数
    real :: ave_gflops = 0.0, max_gflops = 0.0 ! 平均和最大 Gflops
    real :: ave_time = 0.0, min_time = 1e9    ! 平均和最小时间
    logical :: use_double = .false.  ! 默认使用单精度

    integer :: argi, i, dim
    character(len=100) :: arg

    logical :: double_set = .false., float_set = .false.  ! 新增逻辑变量跟踪选项设置

    integer :: my_rank, num_procs, ierr

    ! 初始化 MPI
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, num_procs, ierr)

    ! 帮助信息
    if (command_argument_count() == 0 .and. my_rank == 0) then
        call print_help()
        call MPI_Finalize(ierr)
        stop
    end if

    ! 解析 -n, -l, -float, -double 选项
    argi = 1
    do while (argi <= command_argument_count())
        call get_command_argument(argi, arg)
        if (trim(arg) == '-n' .and. argi + 1 <= command_argument_count()) then
            argi = argi + 1
            call get_command_argument(argi, arg)
            read(arg, *) n
        else if (trim(arg) == '-l' .and. argi + 1 <= command_argument_count()) then
            argi = argi + 1
            call get_command_argument(argi, arg)
            read(arg, *) loop_num
        else if (trim(arg) == '-double') then
            double_set = .true.
            use_double = .true.
        else if (trim(arg) == '-float') then
            float_set = .true.
            use_double = .false.
        else if (trim(arg) == '-h' .or. trim(arg) == '--help') then
            if (my_rank == 0) then
                call print_help()
            end if
            call MPI_Finalize(ierr)
            stop
        end if

        ! 检查是否同时设置了 -double 和 -float
        if (double_set .and. float_set) then
            if (my_rank == 0) then
                print *, "Error: Cannot specify both -double and -float options."
            end if
            call MPI_Finalize(ierr)
            stop
        end if

        argi = argi + 1
    end do

    dim = 2**n

    if (use_double) then
        call perform_double(dim, loop_num, ave_gflops, max_gflops, ave_time, min_time, my_rank, num_procs)
    else
        call perform_float(dim, loop_num, ave_gflops, max_gflops, ave_time, min_time, my_rank, num_procs)
    end if

    if (my_rank == 0) then
        ave_gflops = ave_gflops / loop_num
        ave_time = ave_time / loop_num
        print '(A, F8.3, A, F8.3)', 'Average Gflops: ', ave_gflops, ', Max Gflops: ', max_gflops
        print '(A, F10.6, A, F10.6, A)', 'Average Time: ', ave_time, 's, Min Time: ', min_time, 's'
    end if

    ! 结束 MPI
    call MPI_Finalize(ierr)

contains

    subroutine print_help()
        print *, 'Usage: program_name [-n SIZE] [-l LOOP_NUM] [-float|-double]'
        print *, '  -n SIZE      Specify matrix size, like 2^SIZE (default: 10)'
        print *, '  -l LOOP_NUM  Specify number of iterations (default: 5)'
        print *, '  -float       Use real*32 precision (default)'
        print *, '  -double      Use real*64 precision'
        print *, '  -h, --help   Show this help message'
    end subroutine print_help

    subroutine initialize_matrix_float(n, my_rank, matrix)
        integer, intent(in) :: n, my_rank
        real, intent(out) :: matrix(n, n)  
        integer :: i, j

        call srand(int(time()) + my_rank)
        do i = 1, n
            do j = 1, n
                matrix(i, j) = real(rand())  
            end do
        end do
    end subroutine initialize_matrix_float

    subroutine initialize_matrix_double(n, my_rank, matrix)
        integer, intent(in) :: n, my_rank
        real*8, intent(out) :: matrix(n, n)  
        integer :: i, j

        call srand(int(time()) + my_rank)
        do i = 1, n
            do j = 1, n
                matrix(i, j) = dble(rand())  
            end do
        end do
    end subroutine initialize_matrix_double

    subroutine perform_double(dim, loop_num, ave_gflops, max_gflops, ave_time, min_time, my_rank, num_procs)
        use mpi
        integer, intent(in) :: dim, loop_num, my_rank, num_procs
        real, intent(inout) :: ave_gflops, max_gflops, ave_time
        real, intent(inout) :: min_time
        real*8, allocatable :: a_double(:, :), b_double(:, :), c_double(:, :)  
        real*8, allocatable :: a_local(:, :), c_local(:, :)
        real :: gflops
        integer*8 :: i, start_count(1), end_count(1), count_rate(1)
        real*8 :: elapsed_time, all_elapsed_time
        integer :: rows_per_proc, extra_rows, offset
        integer, allocatable :: sendcounts(:), displs(:) 
        integer :: ierr
        integer :: k 
        integer :: check_indices(2)
        real*8 :: check_value, result_value

        if (my_rank == 0) then
            print *, 'Using real*64 precision for matrix multiplication.'
        end if

        rows_per_proc = dim / num_procs
        extra_rows = dim - rows_per_proc * num_procs

        if (my_rank < extra_rows) then
            offset = my_rank * (rows_per_proc + 1)
            rows_per_proc = rows_per_proc + 1
        else
            offset = extra_rows * (rows_per_proc + 1) + (my_rank - extra_rows) * rows_per_proc
        end if

        ! 所有进程都分配 b_double 内存
        allocate(b_double(dim, dim))
        if (my_rank == 0) then
            allocate(a_double(dim, dim), c_double(dim, dim))  
            call initialize_matrix_double(dim, my_rank, a_double)
            call initialize_matrix_double(dim, my_rank, b_double)

            ! 生成校验值
            call srand(int(time()) + my_rank)
            check_indices(1) = mod(int(rand() * dim), dim) + 1
            check_indices(2) = mod(int(rand() * dim), dim) + 1
            check_value = 0.0d0
            do k = 1, dim
                check_value = check_value + a_double(check_indices(1), k) * b_double(k, check_indices(2))
            end do
        end if

        ! 广播校验行列索引和校验值
        call MPI_Bcast(check_indices, 2, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
        call MPI_Bcast(check_value, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)

        allocate(a_local(rows_per_proc, dim), c_local(rows_per_proc, dim))
        allocate(sendcounts(num_procs), displs(num_procs)) 

        do k = 1, num_procs
            if (k <= extra_rows) then
                sendcounts(k) = (dim / num_procs + 1) * dim
            else
                sendcounts(k) = (dim / num_procs) * dim
            end if
            if (k == 1) then
                displs(k) = 0
            else
                displs(k) = displs(k - 1) + sendcounts(k - 1)
            end if
        end do

        call MPI_Scatterv(a_double, sendcounts, displs, MPI_DOUBLE_PRECISION, &
                          a_local, rows_per_proc * dim, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
        call MPI_Bcast(b_double, dim * dim, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)

        do i = 1, loop_num
            call system_clock(count=start_count(1), count_rate=count_rate(1))
            call matrix_multiply_double(rows_per_proc, a_local, b_double, c_local, dim)
            call system_clock(count=end_count(1))

            elapsed_time = real(end_count(1) - start_count(1)) / real(count_rate(1))

            call MPI_Allreduce(elapsed_time, all_elapsed_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
            all_elapsed_time = all_elapsed_time / num_procs

            gflops = 1e-9 * dim * dim * dim * 2 / all_elapsed_time

            if (my_rank == 0) then
                print '(I8, A, I0, A, I0, A, F10.6, A, F8.3, A)', i, ' : ', dim, ' x ', dim, ' Matrix multiply wall time : ', all_elapsed_time, 's(', gflops, 'Gflops)'
            end if

            ! 收集所有进程的局部结果到主进程
            if (my_rank == 0) then
                c_double = 0.0d0
            end if
            call MPI_Gatherv(c_local, rows_per_proc * dim, MPI_DOUBLE_PRECISION, c_double, sendcounts, displs, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)

            ! 主进程进行校验
            if (my_rank == 0) then
                result_value = c_double(check_indices(1), check_indices(2))
                if (abs(result_value - check_value) > 0.001d0) then
                    print *, 'Verification failed at iteration ', i, ': expected ', check_value, ', got ', result_value
                end if
            end if

            if (my_rank == 0) then
                ave_gflops = ave_gflops + gflops
                max_gflops = max(max_gflops, gflops)
                ave_time = ave_time + all_elapsed_time
                min_time = min(min_time, all_elapsed_time)
            end if
        end do

        if (my_rank == 0) then
            deallocate(a_double, c_double)
        end if
        deallocate(b_double)
        deallocate(a_local, c_local)
        deallocate(sendcounts, displs) 
    end subroutine perform_double

    subroutine perform_float(dim, loop_num, ave_gflops, max_gflops, ave_time, min_time, my_rank, num_procs)
        use mpi
        integer, intent(in) :: dim, loop_num, my_rank, num_procs
        real, intent(inout) :: ave_gflops, max_gflops, ave_time
        real, intent(inout) :: min_time
        real*4, allocatable :: a_float(:, :), b_float(:, :), c_float(:, :)  
        real*4, allocatable :: a_local(:, :), c_local(:, :)
        real :: gflops
        integer*8 :: i, start_count(1), end_count(1), count_rate(1)
        real*8 :: elapsed_time, all_elapsed_time
        integer :: rows_per_proc, extra_rows, offset
        integer, allocatable :: sendcounts(:), displs(:) 
        integer :: ierr
        integer :: k 
        integer :: check_indices(2)
        real*4 :: check_value, result_value

        if (my_rank == 0) then
            print *, 'Using real*32 precision for matrix multiplication.'
        end if

        rows_per_proc = dim / num_procs
        extra_rows = dim - rows_per_proc * num_procs

        if (my_rank < extra_rows) then
            offset = my_rank * (rows_per_proc + 1)
            rows_per_proc = rows_per_proc + 1
        else
            offset = extra_rows * (rows_per_proc + 1) + (my_rank - extra_rows) * rows_per_proc
        end if

        ! 所有进程都分配 b_float 内存
        allocate(b_float(dim, dim))
        if (my_rank == 0) then
            allocate(a_float(dim, dim), c_float(dim, dim))  
            call initialize_matrix_float(dim, my_rank, a_float)
            call initialize_matrix_float(dim, my_rank, b_float)

            ! 生成校验值
            call srand(int(time()) + my_rank)
            check_indices(1) = mod(int(rand() * dim), dim) + 1
            check_indices(2) = mod(int(rand() * dim), dim) + 1
            check_value = 0.0
            do k = 1, dim
                check_value = check_value + a_float(check_indices(1), k) * b_float(k, check_indices(2))
            end do
        end if

        ! 广播校验行列索引和校验值
        call MPI_Bcast(check_indices, 2, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
        call MPI_Bcast(check_value, 1, MPI_REAL, 0, MPI_COMM_WORLD, ierr)

        allocate(a_local(rows_per_proc, dim), c_local(rows_per_proc, dim))
        allocate(sendcounts(num_procs), displs(num_procs)) 

        do k = 1, num_procs
            if (k <= extra_rows) then
                sendcounts(k) = (dim / num_procs + 1) * dim
            else
                sendcounts(k) = (dim / num_procs) * dim
            end if
            if (k == 1) then
                displs(k) = 0
            else
                displs(k) = displs(k - 1) + sendcounts(k - 1)
            end if
        end do

        call MPI_Scatterv(a_float, sendcounts, displs, MPI_REAL, &
                          a_local, rows_per_proc * dim, MPI_REAL, 0, MPI_COMM_WORLD, ierr)
        call MPI_Bcast(b_float, dim * dim, MPI_REAL, 0, MPI_COMM_WORLD, ierr)

        do i = 1, loop_num
            call system_clock(count=start_count(1), count_rate=count_rate(1))
            call matrix_multiply_float(rows_per_proc, a_local, b_float, c_local, dim)
            call system_clock(count=end_count(1))

            elapsed_time = real(end_count(1) - start_count(1)) / real(count_rate(1))

            call MPI_Allreduce(elapsed_time, all_elapsed_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, ierr)
            all_elapsed_time = all_elapsed_time / num_procs

            gflops = 1e-9 * dim * dim * dim * 2 / all_elapsed_time

            if (my_rank == 0) then
                print '(I8, A, I0, A, I0, A, F10.6, A, F8.3, A)', i, ' : ', dim, ' x ', dim, ' Matrix multiply wall time : ', all_elapsed_time, 's(', gflops, 'Gflops)'
            end if

            ! 收集所有进程的局部结果到主进程
            if (my_rank == 0) then
                c_float = 0.0
            end if
            call MPI_Gatherv(c_local, rows_per_proc * dim, MPI_REAL, c_float, sendcounts, displs, MPI_REAL, 0, MPI_COMM_WORLD, ierr)

            ! 主进程进行校验
            if (my_rank == 0) then
                result_value = c_float(check_indices(1), check_indices(2))
                if (abs(result_value - check_value) > 0.001) then
                    print *, 'Verification failed at iteration ', i, ': expected ', check_value, ', got ', result_value
                end if
            end if

            if (my_rank == 0) then
                ave_gflops = ave_gflops + gflops
                max_gflops = max(max_gflops, gflops)
                ave_time = ave_time + all_elapsed_time
                min_time = min(min_time, all_elapsed_time)
            end if
        end do

        if (my_rank == 0) then
            deallocate(a_float, c_float)
        end if
        deallocate(b_float)
        deallocate(a_local, c_local)
        deallocate(sendcounts, displs) 
    end subroutine perform_float

end program main