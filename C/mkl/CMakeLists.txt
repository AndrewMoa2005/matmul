cmake_minimum_required(VERSION 3.13)
string(REGEX MATCHALL "[a-zA-Z]+\ |[a-zA-Z]+$" DIRNAME "${CMAKE_CURRENT_SOURCE_DIR}")
project(${DIRNAME} LANGUAGES C)
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
set(CMAKE_C_STANDARD 11)

set(EXECUTE_FILE_NAME ${PROJECT_NAME}_${CMAKE_C_COMPILER_FRONTEND_VARIANT}_${CMAKE_C_COMPILER_ID}_${CMAKE_C_COMPILER_VERSION})
string(TOLOWER ${EXECUTE_FILE_NAME} EXECUTE_FILE_NAME)
message(STATUS "EXECUTE_FILE_NAME: ${EXECUTE_FILE_NAME}")

set(MKL_LINK static)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Enable MKL
find_package(MKL CONFIG REQUIRED)

set(SRC_LIST
    src/main.c
    src/mkl.c
)

add_executable(${EXECUTE_FILE_NAME} ${SRC_LIST})

# set static linking for MSVC
if(MSVC)
    message(STATUS "Configuring for MSVC static linking")
    set_property(TARGET ${EXECUTE_FILE_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_compile_options(${EXECUTE_FILE_NAME} PUBLIC
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>
)
target_include_directories(${EXECUTE_FILE_NAME} PUBLIC
    $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(${EXECUTE_FILE_NAME} PUBLIC
    OpenMP::OpenMP_C
    $<LINK_ONLY:MKL::MKL>
)

# set installation directories
install(TARGETS ${EXECUTE_FILE_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
