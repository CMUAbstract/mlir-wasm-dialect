/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "am_mcu_apollo.h"
#include "bh_assert.h"
#include "bh_log.h"
#include "bh_platform.h"
#include "input_data.h"
#include "wasm.h"
#include "wasm_export.h"
#include <stdlib.h>
#include <string.h>

#define CONFIG_APP_STACK_SIZE 256000
#define CONFIG_APP_HEAP_SIZE 256000
#define CONFIG_GLOBAL_HEAP_BUF_SIZE WASM_GLOBAL_HEAP_SIZE

void gpio_toggle(wasm_exec_env_t exec_env) {
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
}

void delay(wasm_exec_env_t exec_env, int ms) {
  // am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
}

typedef struct {
  uint32_t base_ptr;
  uint32_t data;
  uint32_t offset;
  uint32_t sizes[3];
  uint32_t strides[3];
} Input;

#define INPUT_TENSOR_SIZE 3136
#define OUTPUT_TENSOR_SIZE 40

typedef struct {
  uint32_t base_ptr;
  uint32_t data;
  uint32_t offset;
  uint32_t sizes[2];
  uint32_t strides[2];
} Output;

static void *app_instance_main(wasm_module_inst_t module_inst) {
  const char *exception;

  Input *input_native_ptr = NULL;
  float32 *input_tensor_native_ptr = NULL;
  Output *output_native_ptr = NULL;
  float32 *output_tensor_native_ptr = NULL;
  uint32 argv[2];

  wasm_exec_env_t exec_env =
      wasm_runtime_create_exec_env(module_inst, CONFIG_APP_STACK_SIZE);
  if (!exec_env) {
    printk("Create exec env failed\n");
    return NULL;
  }

  printf("WASM_EXECUTION_TYPE: %d\n", WASM_EXECUTION_TYPE);

  if (WASM_EXECUTION_TYPE == 0 || WASM_EXECUTION_TYPE == 1) {
    wasm_function_inst_t malloc_fn =
        wasm_runtime_lookup_function(module_inst, "malloc");
    if (!malloc_fn) {
      printk("Fail to find function: malloc\n");
      return NULL;
    }

    argv[0] = INPUT_TENSOR_SIZE;
    wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
    uint32_t input_tensor_ptr = argv[0];

    input_tensor_native_ptr =
        wasm_runtime_addr_app_to_native(module_inst, input_tensor_ptr);

    // preprocess data
    float32 scaled_data[28 * 28];
    for (int i = 0; i < 28 * 28; i++) {
      scaled_data[i] = (float32)input_data[i] / (float32)255.0;
    }
    memcpy(input_tensor_native_ptr, scaled_data, INPUT_TENSOR_SIZE);

    argv[0] = sizeof(Input);
    wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
    uint32_t input_ptr = argv[0];

    input_native_ptr = wasm_runtime_addr_app_to_native(module_inst, input_ptr);
    Input input = {
        .base_ptr = input_tensor_ptr,
        .data = input_tensor_ptr,
        .offset = 0,
        .sizes = {1, 28, 28},
        .strides = {28 * 28, 28, 1},
    };
    memcpy(input_native_ptr, &input, sizeof(Input));

    argv[0] = OUTPUT_TENSOR_SIZE;
    wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
    uint32_t output_tensor_ptr = argv[0];

    argv[0] = sizeof(Output);
    wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
    uint32_t output_ptr = argv[0];

    if (WASM_EXECUTION_TYPE == 0) {
      // load main
      wasm_function_inst_t main_func =
          wasm_runtime_lookup_function(module_inst, "main");
      if (!main_func) {
        printk("Fail to find function: main\n");
        return NULL;
      }
      // setup arguments
      argv[0] = input_tensor_ptr;

      // call main
      am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
      if (!wasm_runtime_call_wasm(exec_env, main_func, 1, argv)) {
        printk("call main fail\n");
      }
      am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);

      if ((exception = wasm_runtime_get_exception(module_inst)))
        printk("%s\n", exception);

      // read and print output
      output_tensor_native_ptr =
          wasm_runtime_addr_app_to_native(module_inst, argv[0]);

      for (int i = 0; i < 10; i++) {
        printk("%d: %f\n", i, output_tensor_native_ptr[i]);
      }
    } else if (WASM_EXECUTION_TYPE == 1) {
      // load main
      wasm_function_inst_t main_func =
          wasm_runtime_lookup_function(module_inst, "_mlir_ciface_main");
      if (!main_func) {
        printk("Fail to find function: _mlir_ciface_main\n");
        return NULL;
      }
      // setup arguments
      argv[0] = output_ptr;
      argv[1] = input_ptr;

      // call main
      am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
      if (!wasm_runtime_call_wasm(exec_env, main_func, 2, argv)) {
        printk("call main fail\n");
      }
      am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);

      if ((exception = wasm_runtime_get_exception(module_inst)))
        printk("%s\n", exception);

      // read and print output
      output_native_ptr =
          wasm_runtime_addr_app_to_native(module_inst, output_ptr);

      Output output;
      memcpy(&output, output_native_ptr, sizeof(Output));
      output_tensor_native_ptr =
          wasm_runtime_addr_app_to_native(module_inst, output.data);

      for (int i = 0; i < 10; i++) {
        printk("%d: %f\n", i, output_tensor_native_ptr[i]);
      }
    }
  } else if (WASM_EXECUTION_TYPE == 2) {
    wasm_function_inst_t main_func =
        wasm_runtime_lookup_function(module_inst, "__main_argc_argv");
    if (!main_func) {
      printk("Fail to find function: __main_argc_argv\n");
      return NULL;
    }

    if (!wasm_runtime_call_wasm(exec_env, main_func, 2, argv)) {
      printk("call main fail\n");
    }

    if ((exception = wasm_runtime_get_exception(module_inst)))
      printk("%s\n", exception);
  }

  wasm_runtime_destroy_exec_env(exec_env);

  return NULL;
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
  app_instance_main(wasm_module_inst);
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
