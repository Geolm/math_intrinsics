cmake_minimum_required(VERSION 3.25)

set(CMAKE_OSX_ARCHITECTURES arm64;x86_64)

project(math_intrinsics_unit_tests)

add_executable(test_precision test.c math_intrinsics.c)
add_executable(test_fast test.c math_intrinsics.c)
add_executable(benchmark_precision benchmark.c math_intrinsics.c)
add_executable(benchmark_fast benchmark.c math_intrinsics.c)

if(LINUX)
    set(CMAKE_EXE_LINKER_FLAGS "-lm")
endif()

if(MSVC)
    target_compile_options(test_precision PRIVATE /W4 /WX /std:c17)
    target_compile_options(test_fast PRIVATE /W4 /WX /std:c17 /D__MATH_INTRINSINCS_FAST__)
    target_compile_options(benchmark_precision PRIVATE /O2 /std:c17)
    target_compile_options(benchmark_fast PRIVATE /O2 /std:c17 /D__MATH_INTRINSINCS_FAST__)
else()
    target_compile_options(test_precision PRIVATE -Wall -Wextra -Wpedantic -Werror -mavx2 -mfma)
    target_compile_options(test_fast PRIVATE -Wall -Wextra -Wpedantic -Werror -mavx2 -mfma -D__MATH_INTRINSINCS_FAST__)
    target_compile_options(benchmark_precision PRIVATE -O3 -mavx2 -mfma)
    target_compile_options(benchmark_fast PRIVATE -O3 -mavx2 -mfma -D__MATH_INTRINSINCS_FAST__)
endif()