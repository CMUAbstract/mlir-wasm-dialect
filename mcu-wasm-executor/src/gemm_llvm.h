#include "utility.h"

#define SIZE 256

static void *app_instance_main(wasm_module_inst_t module_inst,
                               wasm_exec_env_t exec_env) {
  const char *exception;
  uint32_t argv[5];

  float32 data2d[SIZE * SIZE];
  for (int i = 0; i < SIZE * SIZE; i++) {
    data2d[i] = 1.0;
  }

  InputData arg2_input = initialize_input(module_inst, exec_env,
                                          /*tensor_size=*/SIZE * SIZE,
                                          /*dim=*/2, (uint32_t[]){SIZE, SIZE},
                                          (uint32_t[]){SIZE, 1}, data2d);
  InputData arg3_input = initialize_input(module_inst, exec_env,
                                          /*tensor_size=*/SIZE * SIZE,
                                          /*dim=*/2, (uint32_t[]){SIZE, SIZE},
                                          (uint32_t[]){SIZE, 1}, data2d);
  InputData arg4_input = initialize_input(module_inst, exec_env,
                                          /*tensor_size=*/SIZE * SIZE,
                                          /*dim=*/2, (uint32_t[]){SIZE, SIZE},
                                          (uint32_t[]){SIZE, 1}, data2d);

  // load main
  wasm_function_inst_t main_func =
      wasm_runtime_lookup_function(module_inst, "_mlir_ciface_main");
  if (!main_func) {
    printk("Fail to find function: main\n");
    return NULL;
  }
  // setup arguments
  argv[0] = 1.2;
  argv[1] = 1.2;
  argv[2] = arg2_input.input_ptr;
  argv[3] = arg3_input.input_ptr;
  argv[4] = arg4_input.input_ptr;

  // call main
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
  if (!wasm_runtime_call_wasm(exec_env, main_func, 5, argv)) {
    printk("call main fail\n");
  }
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);

  if ((exception = wasm_runtime_get_exception(module_inst)))
    printk("%s\n", exception);

  // read and print output
  for (int i = 0; i < 10; i++) {
    printk("%d: %f\n", i, (double)arg2_input.tensor_native_ptr[i]);
  }

  wasm_runtime_destroy_exec_env(exec_env);
  return NULL;
}