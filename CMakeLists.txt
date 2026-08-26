cmake_minimum_required(VERSION 3.10)

project(MATMUL_CPU C Fortran)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(matmul_compile_flags INTERFACE)
target_compile_options(
    matmul_compile_flags
    INTERFACE -Wall -O3 -march=native -mno-avx512f
)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenBLAS REQUIRED)

add_subdirectory(src)

option(OPENBLAS "Use OPENBLAS library" OFF)
option(OPENMP "Use matmul_openmp implementation" OFF)
option(THREADPOOL "Use matmul_threadpool implementation" OFF)
option(LOCKFREE "Use matmul_lockfree implementation" OFF)

add_executable(test_matmul test.c)
target_link_libraries(test_matmul PRIVATE matmul m OpenBLAS::OpenBLAS)

target_compile_definitions(
    test_matmul
    PRIVATE
        $<$<BOOL:${OPENBLAS}>:OPENBLAS>
        $<$<BOOL:${OPENMP}>:OPENMP>
        $<$<BOOL:${THREADPOOL}>:THREADPOOL>
        $<$<BOOL:${LOCKFREE}>:LOCKFREE>
)

add_executable(benchmark benchmark.c)
target_link_libraries(benchmark PRIVATE matmul m OpenBLAS::OpenBLAS)

target_compile_definitions(
    benchmark
    PRIVATE
        $<$<BOOL:${OPENBLAS}>:OPENBLAS>
        $<$<BOOL:${OPENMP}>:OPENMP>
        $<$<BOOL:${THREADPOOL}>:THREADPOOL>
        $<$<BOOL:${LOCKFREE}>:LOCKFREE>
)

# if(OPENBLAS)
#     target_compile_definitions(benchmark PRIVATE OPENBLAS)
# endif()
