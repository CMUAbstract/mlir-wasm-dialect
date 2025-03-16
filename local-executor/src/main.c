/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "bh_assert.h"
#include "bh_log.h"
#include "bh_platform.h"
#include "wasm.h"
#include "wasm_export.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "polybench.h"

#define CONFIG_APP_STACK_SIZE 10240000
#define CONFIG_APP_HEAP_SIZE 10240000
#define CONFIG_GLOBAL_HEAP_BUF_SIZE WASM_GLOBAL_HEAP_SIZE

#define WARMUP 20
#define ITERATIONS 100
// Array to store the elapsed time (in milliseconds) for ITERATIONS pairs.
static long measurements[ITERATIONS];

// Global variables to track the measurement state.
static int pair_count =
    0;                // Number of completed pairs (first call and second call)
static int phase = 0; // 0 = waiting for first call, 1 = waiting for second call
static struct timespec start_time; // Stores start time for the current pair

// This function should be called 2*(WARMUP + ITERATIONS) times in total.
// It measures the elapsed time between the first and second call of each pair.
// The first WARMUP pairs are ignored; for the remaining ITERATIONS pairs,
// the elapsed time is stored in the measurements array.
void toggle_gpio(wasm_exec_env_t exec_env) {
  struct timespec current_time;

  // Get the current time using a monotonic clock.
  if (clock_gettime(CLOCK_MONOTONIC, &current_time) != 0) {
    perror("clock_gettime");
    return;
  }

  if (phase == 0) {
    // First call of the pair: record the start time.
    start_time = current_time;
    phase = 1;
  } else {
    // Second call: compute elapsed time.
    long sec_diff = current_time.tv_sec - start_time.tv_sec;
    long nsec_diff = current_time.tv_nsec - start_time.tv_nsec;
    if (nsec_diff < 0) {
      sec_diff--;
      nsec_diff += 1000000000L;
    }
    // Convert elapsed time to milliseconds.
    long elapsed_ms = sec_diff * 1000 + nsec_diff / 1000000;

    // For warmup pairs, do nothing; for iterations, store the measurement.
    if (pair_count >= WARMUP && pair_count < (WARMUP + ITERATIONS)) {
      measurements[pair_count - WARMUP] = elapsed_ms;
    }
    pair_count++; // One pair is completed.
    phase = 0;    // Reset phase for the next pair.
  }
}

// This function computes and prints timing statistics for the ITERATIONS
// measurements. It calculates the mean, minimum, maximum, and standard
// deviation.
void print_timing_statistics() {
  if (ITERATIONS <= 0) {
    printf("No iterations to report.\n");
    return;
  }

  long sum = 0;
  long min = measurements[0];
  long max = measurements[0];
  for (int i = 0; i < ITERATIONS; i++) {
    long t = measurements[i];
    sum += t;
    if (t < min) {
      min = t;
    }
    if (t > max) {
      max = t;
    }
  }
  double mean = (double)sum / ITERATIONS;

  // Calculate standard deviation.
  double sum_sq_diff = 0.0;
  for (int i = 0; i < ITERATIONS; i++) {
    double diff = measurements[i] - mean;
    sum_sq_diff += diff * diff;
  }
  double stddev = sqrt(sum_sq_diff / ITERATIONS);

  printf("Timing statistics over %d iterations:\n", ITERATIONS);
  printf("[execution time] %.2f miliseconds\n", mean);
  printf("Min: %ld ms\n", min);
  printf("Max: %ld ms\n", max);
  printf("Standard Deviation: %.2f ms\n", stddev);
}

void print_i32(wasm_exec_env_t exec_env, int32_t i) {
  // print only first 10 times
  static int print_count = 0;
  if (print_count < 10) {
    printf("%d\n", i);
    print_count++;
  }
}

#if WASM_ENABLE_GLOBAL_HEAP_POOL != 0
static char global_heap_buf[CONFIG_GLOBAL_HEAP_BUF_SIZE] = {0};
#endif

void iwasm_main() {
  uint8 *wasm_file_buf = NULL;
  uint32 wasm_file_size;
  wasm_module_t wasm_module = NULL;
  wasm_module_inst_t wasm_module_inst = NULL;
  RuntimeInitArgs init_args;
  char error_buf[128];

#if WASM_ENABLE_LOG != 0
  int log_verbose_level = 2;
#endif

  memset(&init_args, 0, sizeof(RuntimeInitArgs));

#if WASM_ENABLE_GLOBAL_HEAP_POOL != 0
  init_args.mem_alloc_type = Alloc_With_Pool;
  init_args.mem_alloc_option.pool.heap_buf = global_heap_buf;
  init_args.mem_alloc_option.pool.heap_size = sizeof(global_heap_buf);
#elif (defined(CONFIG_COMMON_LIBC_MALLOC) &&                                   \
       CONFIG_COMMON_LIBC_MALLOC_ARENA_SIZE != 0) ||                           \
    defined(CONFIG_NEWLIB_LIBC)
  init_args.mem_alloc_type = Alloc_With_System_Allocator;
#else
#error "memory allocation scheme is not defined."
#endif

  /* initialize runtime environment */
  if (!wasm_runtime_full_init(&init_args)) {
    printf("Init runtime environment failed.\n");
    return;
  }

#if WASM_ENABLE_LOG != 0
  bh_log_set_verbose_level(log_verbose_level);
#endif

  /* register native symbols */
  static NativeSymbol native_symbols[] = {{"toggle_gpio", toggle_gpio, "()"},
                                          {"print_i32", print_i32, "(i)"}};
  int n_native_symbols = sizeof(native_symbols) / sizeof(NativeSymbol);

  if (!wasm_runtime_register_natives("env", native_symbols, n_native_symbols)) {
    printf("Register natives failed.\n");
    goto fail1;
  }

  /* load WASM byte buffer from byte buffer of include file */
  wasm_file_buf = (uint8 *)wasm_file;
  wasm_file_size = wasm_file_len;

  printf("wasm file size: %d\n", wasm_file_size);

  /* load WASM module */
  if (!(wasm_module = wasm_runtime_load(wasm_file_buf, wasm_file_size,
                                        error_buf, sizeof(error_buf)))) {
    printf("%s\n", error_buf);
    goto fail1;
  }

  printf("heap size: %d\n", CONFIG_APP_HEAP_SIZE);
  printf("stack size: %d\n", CONFIG_APP_STACK_SIZE);

  for (int i = 0; i < WARMUP + ITERATIONS; i++) {
    /* instantiate the module */
    if (!(wasm_module_inst = wasm_runtime_instantiate(
              wasm_module, CONFIG_APP_STACK_SIZE, CONFIG_APP_HEAP_SIZE,
              error_buf, sizeof(error_buf)))) {
      printf("%s\n", error_buf);
      goto fail2;
    }

    wasm_exec_env_t exec_env =
        wasm_runtime_create_exec_env(wasm_module_inst, CONFIG_APP_STACK_SIZE);
    if (!exec_env) {
      printf("Create exec env failed\n");
      goto fail2;
    }

    app_instance_main(wasm_module_inst, exec_env);
  }

  print_timing_statistics();

  /* destroy the module instance */
  wasm_runtime_deinstantiate(wasm_module_inst);

fail2:
  /* unload the module */
  wasm_runtime_unload(wasm_module);

fail1:
  /* destroy runtime environment */
  wasm_runtime_destroy();
}

int main(void) {
  iwasm_main();
  return 0;
}
