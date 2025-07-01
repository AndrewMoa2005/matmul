program main
    implicit none
    external matrix_multiply_float, matrix_multiply_double
    integer :: n = 10         
    integer :: loop_num = 5   
    real :: ave_gflops = 0.0, max_gflops = 0.0 
    real :: ave_time = 0.0, min_time = 1e9    
    logical :: use_double = .false.  

    integer :: argi, i, dim
    character(len=100) :: arg

    logical :: double_set = .false., float_set = .false.  

    if (command_argument_count() == 0) then
        call print_help()
        stop
    end if

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
            call print_help()
            stop
        end if

        if (double_set .and. float_set) then
            print *, "Error: Cannot specify both -double and -float options."
            stop
        end if

        argi = argi + 1
    end do

    dim = 2**n

    if (use_double) then
        call perform_double(dim, loop_num, ave_gflops, max_gflops, ave_time, min_time)
    else
        call perform_float(dim, loop_num, ave_gflops, max_gflops, ave_time, min_time)
    end if

    ave_gflops = ave_gflops / loop_num
    ave_time = ave_time / loop_num
    print '(A, F8.3, A, F8.3)', 'Average Gflops: ', ave_gflops, ', Max Gflops: ', max_gflops
    print '(A, F10.6, A, F10.6, A)', 'Average Time: ', ave_time, 's, Min Time: ', min_time, 's'

contains

    subroutine print_help()
        print *, 'Usage: program_name [-n SIZE] [-l LOOP_NUM] [-float|-double]'
        print *, '  -n SIZE      Specify matrix size, like 2^SIZE (default: 10)'
        print *, '  -l LOOP_NUM  Specify number of iterations (default: 5)'
        print *, '  -float       Use real*32 precision (default)'
        print *, '  -double      Use real*64 precision'
        print *, '  -h, --help   Show this help message'
    end subroutine print_help

    subroutine initialize_matrix_float(n, matrix)
        integer, intent(in) :: n
        real, intent(out) :: matrix(n, n)  
        integer :: i, j
        real :: rand

        call random_seed()
        do i = 1, n
            do j = 1, n
                call random_number(rand)  
                matrix(i, j) = rand
            end do
        end do
    end subroutine initialize_matrix_float

    subroutine initialize_matrix_double(n, matrix)
        integer, intent(in) :: n
        real*8, intent(out) :: matrix(n, n)  
        integer :: i, j
        real*8 :: rand

        call random_seed()
        do i = 1, n
            do j = 1, n
                call random_number(rand)  
                matrix(i, j) = rand
            end do
        end do
    end subroutine initialize_matrix_double

    subroutine perform_double(dim, loop_num, ave_gflops, max_gflops, ave_time, min_time)
        integer, intent(in) :: dim, loop_num
        real, intent(inout) :: ave_gflops, max_gflops, ave_time
        real, intent(inout) :: min_time
        real*8, allocatable :: a_double(:, :), b_double(:, :), c_double(:, :) 
        real :: gflops
        integer*8 :: i, start_count(1), end_count(1), count_rate(1)
        real :: elapsed_time

        print *, 'Using real*64 precision for matrix multiplication.'
        allocate(a_double(dim, dim), b_double(dim, dim), c_double(dim, dim)) 
        do i = 1, loop_num
            call initialize_matrix_double(dim, a_double)
            call initialize_matrix_double(dim, b_double)

            call system_clock(count=start_count(1), count_rate=count_rate(1))
            call matrix_multiply_double(dim, a_double, b_double, c_double)
            call system_clock(count=end_count(1))

            elapsed_time = real(end_count(1) - start_count(1)) / real(count_rate(1))
            gflops = 1e-9 * dim * dim * dim * 2 / elapsed_time
            print '(I8, A, I0, A, I0, A, F10.6, A, F8.3, A)', i, ' : ', dim, ' x ', dim, ' Matrix multiply wall time : ', elapsed_time, 's(', gflops, 'Gflops)'

            ave_gflops = ave_gflops + gflops
            max_gflops = max(max_gflops, gflops)
            ave_time = ave_time + elapsed_time
            min_time = min(min_time, elapsed_time)
        end do
        deallocate(a_double, b_double, c_double)
    end subroutine perform_double

    subroutine perform_float(dim, loop_num, ave_gflops, max_gflops, ave_time, min_time)
        integer, intent(in) :: dim, loop_num
        real, intent(inout) :: ave_gflops, max_gflops, ave_time
        real, intent(inout) :: min_time
        real*4, allocatable :: a_float(:, :), b_float(:, :), c_float(:, :)  
        real :: gflops
        integer*8 :: i, start_count(1), end_count(1), count_rate(1)
        real :: elapsed_time

        print *, 'Using real*32 precision for matrix multiplication.'
        allocate(a_float(dim, dim), b_float(dim, dim), c_float(dim, dim))  
        do i = 1, loop_num
            call initialize_matrix_float(dim, a_float)
            call initialize_matrix_float(dim, b_float)

            call system_clock(count=start_count(1), count_rate=count_rate(1))
            call matrix_multiply_float(dim, a_float, b_float, c_float)
            call system_clock(count=end_count(1))

            elapsed_time = real(end_count(1) - start_count(1)) / real(count_rate(1))
            gflops = 1e-9 * dim * dim * dim * 2 / elapsed_time
            print '(I8, A, I0, A, I0, A, F10.6, A, F8.3, A)', i, ' : ', dim, ' x ', dim, ' Matrix multiply wall time : ', elapsed_time, 's(', gflops, 'Gflops)'

            ave_gflops = ave_gflops + gflops
            max_gflops = max(max_gflops, gflops)
            ave_time = ave_time + elapsed_time
            min_time = min(min_time, elapsed_time)
        end do
        deallocate(a_float, b_float, c_float)
    end subroutine perform_float

end program main
