# Copyright (C) 2019 Intel Corporation.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.8.2)

set(WAMR_ROOT_DIR $ENV{WAMR_ROOT_DIR})

# Check if WAMR_ROOT_DIR was set
if(NOT WAMR_ROOT_DIR)
    message(FATAL_ERROR "Environment variable 'WAMR_ROOT_DIR' is not defined. Please set it before configuring.")
endif()

project(wamr)

enable_language(ASM)

# Update this based on the platform you are running on
set(WAMR_BUILD_PLATFORM "darwin")

set(WAMR_BUILD_INTERP 1)
set(WAMR_BUILD_JIT 0)
set(WAMR_BUILD_AOT 1)
set(WAMR_BUILD_FAST_JIT 0)
set(WAMR_BUILD_LIBC_BUILTIN 1)
set(WAMR_BUILD_LIBC_WASI 0)
set(WAMR_BUILD_LIB_PTHREAD 0)
set(WAMR_BUILD_GLOBAL_HEAP_POOL 1)
set(WAMR_BUILD_GLOBAL_HEAP_SIZE 32778420)

include(${WAMR_ROOT_DIR}/build-scripts/runtime_lib.cmake)
include(${WAMR_ROOT_DIR}/core/shared/platform/common/math/platform_api_math.cmake)

add_executable(app src/main.c)


# Add the new include directory for the moved headers
target_include_directories(app PRIVATE ../polybench/src)

target_sources(app PRIVATE
               ${WAMR_RUNTIME_LIB_SOURCE}
               src/main.c)

target_sources(app PRIVATE ../polybench/src/polybench.h)

target_compile_options(app PRIVATE -O3)