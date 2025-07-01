cmake_minimum_required(VERSION 3.13)
get_filename_component(CURRENT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
get_filename_component(FOLDER_NAME ${CURRENT_DIR} NAME)
message(STATUS "FOLDER_NAME: ${FOLDER_NAME}")
project(${FOLDER_NAME} LANGUAGES C)
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
set(CMAKE_C_STANDARD 11)

get_filename_component(FILE_NAME ${CMAKE_C_COMPILER} NAME)
string(REGEX REPLACE ".exe" "" COMPILER_TMP "${FILE_NAME}")
string(REGEX REPLACE "-" "" COMPILER "${COMPILER_TMP}")
message(STATUS "COMPILER: ${COMPILER}")
set(EXECUTE_FILE_NAME ${PROJECT_NAME}_${CMAKE_C_COMPILER_FRONTEND_VARIANT}_${COMPILER}_${CMAKE_C_COMPILER_VERSION})
string(TOLOWER ${EXECUTE_FILE_NAME} EXECUTE_FILE_NAME)
message(STATUS "EXECUTE_FILE_NAME: ${EXECUTE_FILE_NAME}")

# setting aocl directory
if(DEFINED ROCM_DIR)
    message(STATUS "ROCM_DIR is set to: ${ROCM_DIR}")
else()
    message(FATAL_ERROR "ROCM_DIR is not defined. Please set it to the AOCL installation directory.")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__HIP_PLATFORM_AMD__")

# Enable AOCL BLAS
set(ROCM_INCLUDE_DIRS ${ROCM_DIR}/include)
set(ROCM_LINK_DIR ${ROCM_DIR}/lib)
set(ROCM_LIBS "rocblas;amdhip64")

set(SRC_LIST
    src/main.c
    src/blas.c
)

add_executable(${EXECUTE_FILE_NAME} ${SRC_LIST})

target_include_directories(${EXECUTE_FILE_NAME} PRIVATE
    ${ROCM_INCLUDE_DIRS}
)
target_link_directories(${EXECUTE_FILE_NAME} PRIVATE
    ${ROCM_LINK_DIR}
)

# set static linking for MSVC
if(MSVC)
    message(STATUS "Configuring for MSVC static linking")
    set_property(TARGET ${EXECUTE_FILE_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_link_libraries(${EXECUTE_FILE_NAME} PRIVATE
    ${ROCM_LIBS}
)

# set installation directories
install(TARGETS ${EXECUTE_FILE_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
