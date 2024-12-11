/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "am_mcu_apollo.h"
#include "bh_assert.h"
#include "bh_log.h"
#include "bh_platform.h"
#include "wasm.h"
#include "wasm_export.h"
#include <stdlib.h>
#include <string.h>

#if defined(MNIST_MLIR)
#include "mnist_mlir.h"
#elif defined(MNIST_LLVM)
#include "mnist_llvm.h"
#elif defined(ATAX_MLIR)
#include "atax_mlir.h"
#elif defined(ATAX_LLVM)
#include "atax_llvm.h"
#elif defined(BICG_MLIR)
#include "bicg_mlir.h"
#elif defined(BICG_LLVM)
#include "bicg_llvm.h"
#elif defined(DOITGEN_MLIR)
#include "doitgen_mlir.h"
#elif defined(DOITGEN_LLVM)
#include "doitgen_llvm.h"
#elif defined(GEMM_MLIR)
#include "gemm_mlir.h"
#elif defined(GEMM_LLVM)
#include "gemm_llvm.h"
#elif defined(GEMVER_MLIR)
#include "gemver_mlir.h"
#elif defined(GEMVER_LLVM)
#include "gemver_llvm.h"
#elif defined(GESUMMV_MLIR)
#include "gesummv_mlir.h"
#elif defined(GESUMMV_LLVM)
#include "gesummv_llvm.h"
#elif defined(MVT_MLIR)
#include "mvt_mlir.h"
#elif defined(MVT_LLVM)
#include "mvt_llvm.h"
#elif defined(SYMM_MLIR)
#include "symm_mlir.h"
#elif defined(SYMM_LLVM)
#include "symm_llvm.h"
#elif defined(SYR2K_MLIR)
#include "syr2k_mlir.h"
#elif defined(SYR2K_LLVM)
#include "syr2k_llvm.h"
#elif defined(THREE_MM_MLIR)
#include "three_mm_mlir.h"
#elif defined(THREE_MM_LLVM)
#include "three_mm_llvm.h"
#elif defined(TRMM_MLIR)
#include "trmm_mlir.h"
#elif defined(TRMM_LLVM)
#include "trmm_llvm.h"
#elif defined(TWO_MM_MLIR)
#include "two_mm_mlir.h"
#elif defined(TWO_MM_LLVM)
#include "two_mm_llvm.h"
#endif

#define CONFIG_APP_STACK_SIZE 256000
#define CONFIG_APP_HEAP_SIZE 256000
#define CONFIG_GLOBAL_HEAP_BUF_SIZE WASM_GLOBAL_HEAP_SIZE

void gpio_toggle(wasm_exec_env_t exec_env) {
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
}

void delay(wasm_exec_env_t exec_env, int ms) {
  // am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
}

#if WASM_ENABLE_GLOBAL_HEAP_POOL != 0
static char global_heap_buf[CONFIG_GLOBAL_HEAP_BUF_SIZE] = {0};
#endif

void iwasm_main() {
  int start, module_init, module_load, finish_main, end;
  uint8 *wasm_file_buf = NULL;
  uint32 wasm_file_size;
  wasm_module_t wasm_module = NULL;
  wasm_module_inst_t wasm_module_inst = NULL;
  RuntimeInitArgs init_args;
  char error_buf[128];

  start = k_uptime_get_32();

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
    printk("Init runtime environment failed.\n");
    return;
  }

#if WASM_ENABLE_LOG != 0
  bh_log_set_verbose_level(log_verbose_level);
#endif

  /* register native symbols */
  static NativeSymbol native_symbols[] = {{"gpio_toggle", gpio_toggle, "()"

                                          },
                                          {
                                              "delay",
                                              delay,
                                              "(i)",
                                          }};
  int n_native_symbols = sizeof(native_symbols) / sizeof(NativeSymbol);

  if (!wasm_runtime_register_natives("env", native_symbols, n_native_symbols)) {
    printf("Register natives failed.\n");
    goto fail1;
  }

  /* load WASM byte buffer from byte buffer of include file */
  wasm_file_buf = (uint8 *)wasm_file;
  wasm_file_size = wasm_file_len;

  printk("wasm file size: %d\n", wasm_file_size);

  /* load WASM module */
  if (!(wasm_module = wasm_runtime_load(wasm_file_buf, wasm_file_size,
                                        error_buf, sizeof(error_buf)))) {
    printk("%s\n", error_buf);
    goto fail1;
  }

  module_load = k_uptime_get_32();
  printk("elapsed (module load): %d\n", (module_load - start));

  printk("heap size: %d\n", CONFIG_APP_HEAP_SIZE);
  printk("stack size: %d\n", CONFIG_APP_STACK_SIZE);
  printk("clock frequency: %d\n", sys_clock_hw_cycles_per_sec());

  /* instantiate the module */
  if (!(wasm_module_inst = wasm_runtime_instantiate(
            wasm_module, CONFIG_APP_STACK_SIZE, CONFIG_APP_HEAP_SIZE, error_buf,
            sizeof(error_buf)))) {
    printk("%s\n", error_buf);
    goto fail2;
  }

  module_init = k_uptime_get_32();
  printk("elapsed (module instantiation): %d\n", (module_init - module_load));
  /* invoke the main function */

  /* pin 23 measures the time between app instance main */
  am_hal_gpio_state_write(23, AM_HAL_GPIO_OUTPUT_TOGGLE);

  wasm_exec_env_t exec_env =
      wasm_runtime_create_exec_env(wasm_module_inst, CONFIG_APP_STACK_SIZE);
  if (!exec_env) {
    printk("Create exec env failed\n");
    goto fail2;
  }

  app_instance_main(wasm_module_inst, exec_env);
  am_hal_gpio_state_write(23, AM_HAL_GPIO_OUTPUT_TOGGLE);

  finish_main = k_uptime_get_32();
  printk("elapsed (main execution): %d\n", (finish_main - module_init));

  /* destroy the module instance */
  wasm_runtime_deinstantiate(wasm_module_inst);

fail2:
  /* unload the module */
  wasm_runtime_unload(wasm_module);

fail1:
  /* destroy runtime environment */
  wasm_runtime_destroy();

  end = k_uptime_get_32();

  printk("elapsed: %d\n", (end - start));
}

int main(void) {
  am_hal_cachectrl_config_t am_hal_cachectrl_user = {
      .bLRU = 0,
      .eDescript = AM_HAL_CACHECTRL_DESCR_1WAY_128B_4096E,
      .eMode = AM_HAL_CACHECTRL_CONFIG_MODE_INSTR_DATA,
  };

  am_hal_cachectrl_config(&am_hal_cachectrl_user);
  am_hal_cachectrl_enable();

  uint32_t status;

  am_hal_pwrctrl_low_power_init();
  status =
      am_hal_pwrctrl_mcu_mode_select(AM_HAL_PWRCTRL_MCU_MODE_HIGH_PERFORMANCE);

  if (status == AM_HAL_STATUS_SUCCESS) {
    printk("MCU mode selected successfully\n");
  } else {
    printk("Failed to select MCU mode: 0x%08x\n", status);
  }

  //
  // Initialize GPIOs
  //
  am_hal_gpio_pincfg_t am_hal_gpio_pincfg_output = AM_HAL_GPIO_PINCFG_OUTPUT;
  am_hal_gpio_pinconfig(22, am_hal_gpio_pincfg_output);
  am_hal_gpio_pinconfig(23, am_hal_gpio_pincfg_output);

  iwasm_main();
  while (1) {
    //
    // Go to Deep Sleep.
    //

    am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
  }
  return 0;
}
