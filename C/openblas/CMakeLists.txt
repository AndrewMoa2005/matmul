cmake_minimum_required(VERSION 3.13)
string(REGEX MATCHALL "[a-zA-Z]+\ |[a-zA-Z]+$" DIRNAME "${CMAKE_CURRENT_SOURCE_DIR}")
project(${DIRNAME} LANGUAGES C)
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
set(CMAKE_C_STANDARD 11)

set(EXECUTE_FILE_NAME ${PROJECT_NAME}_${CMAKE_C_COMPILER_FRONTEND_VARIANT}_${CMAKE_C_COMPILER_ID}_${CMAKE_C_COMPILER_VERSION})
string(TOLOWER ${EXECUTE_FILE_NAME} EXECUTE_FILE_NAME)
message(STATUS "EXECUTE_FILE_NAME: ${EXECUTE_FILE_NAME}")

# Enable OpenBLAS
find_package(OpenMP REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)

if(NOT OpenBLAS_FOUND)
    message(FATAL_ERROR "OpenBLAS not found")
else()
    message(STATUS "OpenBLAS found: ${OpenBLAS_VERSION}")
    message(STATUS "OpenBLAS include dirs: ${OpenBLAS_INCLUDE_DIRS}")
endif()

set(SRC_LIST
    src/main.c
    src/openblas.c
)

add_executable(${EXECUTE_FILE_NAME} ${SRC_LIST})

# set static linking for MSVC
if(MSVC)
    message(STATUS "Configuring for MSVC static linking")
    set_property(TARGET ${EXECUTE_FILE_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_link_libraries(${EXECUTE_FILE_NAME} PRIVATE
    OpenBLAS::OpenBLAS
    OpenMP::OpenMP_C
)

# set installation directories
install(TARGETS ${EXECUTE_FILE_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
