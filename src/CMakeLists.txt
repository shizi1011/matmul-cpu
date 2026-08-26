if((NOT DEFINED NTHREADS) OR (NTHREADS STREQUAL ""))
    # message(FATAL_ERROR "Please specify -DNTHREADS=<number>")
    message(STATUS "Default number of threads set to 8")
    set(NTHREADS 8)
endif()

# add_library(matmul_threadpool threadpool.c matmul_threadpool.c)
# target_include_directories(matmul_threadpool PUBLIC "${PROJECT_SOURCE_DIR}/src")
# target_link_libraries(matmul_threadpool PRIVATE matmul_compile_flags)
# target_compile_definitions(matmul_threadpool PRIVATE NTHREADS=${NTHREADS})
#
# add_library(matmul_openmp matmul_openmp.c)
# target_include_directories(matmul_openmp PUBLIC "${PROJECT_SOURCE_DIR}/src")
# target_link_libraries(
#     matmul_openmp
#     PRIVATE matmul_compile_flags OpenMP::OpenMP_C
# )
# target_compile_definitions(matmul_openmp PRIVATE NTHREADS=${NTHREADS})

add_library(
    matmul
    matmul_openmp.c
    matmul_naive.c
    matmul_threadpool.c
    matmul_lockfree.c
    threadpool.c
    kernel_utils.c
    utils.c
)
target_include_directories(matmul PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(matmul PRIVATE matmul_compile_flags OpenMP::OpenMP_C)
target_compile_definitions(matmul PRIVATE NTHREADS=${NTHREADS})

# add_library(matmul_lockfree matmul_lockfree.c kernel_utils.c)
# target_include_directories(matmul_lockfree PUBLIC "${PROJECT_SOURCE_DIR}/src")
# target_link_libraries(matmul_lockfree PRIVATE matmul_compile_flags)
# target_compile_definitions(matmul_lockfree PRIVATE NTHREADS=${NTHREADS})
