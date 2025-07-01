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
if(DEFINED AOCL_DIR)
    message(STATUS "AOCL_DIR is set to: ${AOCL_DIR}")
else()
    message(FATAL_ERROR "AOCL_DIR is not defined. Please set it to the AOCL installation directory.")
endif()

if(NOT DEFINED AOCL_BLAS_DATA)
    set(AOCL_BLAS_DATA LP64)
    message(STATUS "AOCL_BLAS_DATA is not defined. Use default value: ${AOCL_BLAS_DATA}")
else()
    message(STATUS "AOCL_BLAS_DATA is set to: ${AOCL_BLAS_DATA}")
endif()

# Enable AOCL BLAS
set(AOCL_BLIS_INCLUDE_DIRS ${AOCL_DIR}/amd-blis/include/${AOCL_BLAS_DATA})
set(AOCL_BLIS_LINK_DIR ${AOCL_DIR}/amd-blis/lib/${AOCL_BLAS_DATA})
set(AOCL_BLIS_LIBS AOCL-LibBlis-Win-MT.lib)

# find OpenMP
find_package(OpenMP REQUIRED)

set(SRC_LIST
    src/main.c
    src/blas.c
)

add_executable(${EXECUTE_FILE_NAME} ${SRC_LIST})

target_include_directories(${EXECUTE_FILE_NAME} PRIVATE
    ${AOCL_BLIS_INCLUDE_DIRS}
)
target_link_directories(${EXECUTE_FILE_NAME} PRIVATE
    ${AOCL_BLIS_LINK_DIR}
)

# set static linking for MSVC
if(MSVC)
    message(STATUS "Configuring for MSVC static linking")
    set_property(TARGET ${EXECUTE_FILE_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_link_libraries(${EXECUTE_FILE_NAME} PRIVATE
    ${AOCL_BLIS_LIBS}
    OpenMP::OpenMP_C
)

# set installation directories
install(TARGETS ${EXECUTE_FILE_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
