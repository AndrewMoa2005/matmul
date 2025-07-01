cmake_minimum_required(VERSION 3.13)
get_filename_component(CURRENT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
get_filename_component(FOLDER_NAME ${CURRENT_DIR} NAME)
message(STATUS "FOLDER_NAME: ${FOLDER_NAME}")
project(${FOLDER_NAME} LANGUAGES Fortran)
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
set(CMAKE_Fortran_STANDARD 2008)

get_filename_component(FILE_NAME ${CMAKE_Fortran_COMPILER} NAME)
string(REGEX REPLACE ".exe" "" COMPILER_TMP "${FILE_NAME}")
string(REGEX REPLACE "-" "" COMPILER "${COMPILER_TMP}")
message(STATUS "COMPILER: ${COMPILER}")
set(EXECUTE_FILE_NAME ${PROJECT_NAME}_${CMAKE_Fortran_COMPILER_FRONTEND_VARIANT}_${COMPILER}_${CMAKE_Fortran_COMPILER_VERSION})
string(TOLOWER ${EXECUTE_FILE_NAME} EXECUTE_FILE_NAME)
message(STATUS "EXECUTE_FILE_NAME: ${EXECUTE_FILE_NAME}")

# Enable MPI support
find_package(MPI REQUIRED)

if(MPI_FOUND)
    message(STATUS "MPI_Fortran_FOUND=${MPI_Fortran_FOUND}")
    message(STATUS "MPI_Fortran_INCLUDE_DIRS=${MPI_Fortran_INCLUDE_DIRS}")
    message(STATUS "MPI_Fortran_LIBRARIES=${MPI_Fortran_LIBRARIES}")
endif()

set(SRC_LIST
    src/main.f90
    src/mpi.f90
)
add_executable(${EXECUTE_FILE_NAME} ${SRC_LIST})

target_include_directories(${EXECUTE_FILE_NAME}
    PRIVATE
    ${MPI_Fortran_INCLUDE_DIRS}
)

# set static linking for MSVC
if(MSVC)
    message(STATUS "Configuring for MSVC static linking")
    set_property(TARGET ${EXECUTE_FILE_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_link_libraries(${EXECUTE_FILE_NAME} PRIVATE
    ${MPI_Fortran_LIBRARIES}
)

# set installation directories
install(TARGETS ${EXECUTE_FILE_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
