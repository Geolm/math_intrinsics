cmake_minimum_required(VERSION 3.25)

set(CMAKE_OSX_ARCHITECTURES arm64;x86_64)

project(math_intrinsics_unit_tests)

add_executable(tests tests.c)

if(LINUX)
    set(CMAKE_EXE_LINKER_FLAGS "-lm")
endif()

if(MSVC)
    target_compile_options(tests PRIVATE /W4 /WX /std:c17)
else()
    target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic -Werror -mavx2 -mfma)
endif()