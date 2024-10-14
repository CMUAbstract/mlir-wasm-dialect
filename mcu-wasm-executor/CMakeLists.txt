# Copyright (C) 2019 Intel Corporation.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.8.2)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
set (WAMR_ROOT_DIR /Users/byeongje/wasm/wasm-micro-runtime)

project(wamr)

enable_language (ASM)

set (WAMR_BUILD_PLATFORM "zephyr")

set (WAMR_BUILD_TARGET "THUMBV7_VFP")
set (WAMR_BUILD_INTERP 1)
set (WAMR_BUILD_JIT 0)
set (WAMR_BUILD_AOT 1)
set (WAMR_BUILD_FAST_JIT 0)
set (WAMR_BUILD_LIBC_BUILTIN 1)
set (WAMR_BUILD_LIBC_WASI 0)
set (WAMR_BUILD_LIB_PTHREAD 0)
set (WAMR_BUILD_GLOBAL_HEAP_POOL 1)
set (WAMR_BUILD_GLOBAL_HEAP_SIZE 1638912) 



add_compile_definitions(WASM_EXECUTION_TYPE=${EXECUTION_TYPE})

include (${WAMR_ROOT_DIR}/build-scripts/runtime_lib.cmake)
include (${WAMR_ROOT_DIR}/core/shared/platform/common/math/platform_api_math.cmake)

zephyr_library_include_directories(${ZEPHYR_BASE}/../modules/hal/ambiq)

target_sources(app PRIVATE
               ${WAMR_RUNTIME_LIB_SOURCE}
               src/main.c)

target_compile_options(app PRIVATE -static -mthumb -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O3)