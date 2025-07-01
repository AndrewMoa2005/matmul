cmake_minimum_required(VERSION 3.13)
string(REGEX MATCHALL "[a-zA-Z]+\ |[a-zA-Z]+$" DIRNAME "${CMAKE_CURRENT_SOURCE_DIR}")
project(${DIRNAME} LANGUAGES C Fortran)
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
set(CMAKE_C_STANDARD 11)
set(CMAKE_Fortran_STANDARD 2008)

set(EXECUTE_FILE_NAME ${PROJECT_NAME}_${CMAKE_Fortran_COMPILER_FRONTEND_VARIANT}_${CMAKE_Fortran_COMPILER_ID}_${CMAKE_Fortran_COMPILER_VERSION})
string(TOLOWER ${EXECUTE_FILE_NAME} EXECUTE_FILE_NAME)
message(STATUS "EXECUTE_FILE_NAME: ${EXECUTE_FILE_NAME}")

# Enable MPI support
find_package(MPI REQUIRED)

if(MPI_FOUND)
    message(STATUS "MPI_C_FOUND=${MPI_C_FOUND}")
    message(STATUS "MPI_C_INCLUDE_DIRS=${MPI_C_INCLUDE_DIRS}")
    message(STATUS "MPI_LIBRARIES=${MPI_LIBRARIES}")
endif()

set(SRC_LIST
    src/main.c
    src/mpi2c.f90
)
add_executable(${EXECUTE_FILE_NAME} ${SRC_LIST})

target_include_directories(${EXECUTE_FILE_NAME}
    PRIVATE
    ${MPI_C_INCLUDE_DIRS}
)

# set static linking for MSVC
if(MSVC)
    message(STATUS "Configuring for MSVC static linking")
    set_property(TARGET ${EXECUTE_FILE_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_link_libraries(${EXECUTE_FILE_NAME} PRIVATE
    ${MPI_LIBRARIES}
)

# set installation directories
install(TARGETS ${EXECUTE_FILE_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
