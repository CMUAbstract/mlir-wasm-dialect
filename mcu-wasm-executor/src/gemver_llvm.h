#include "utility.h"

#define SIZE 256

static void *app_instance_main(wasm_module_inst_t module_inst,
                               wasm_exec_env_t exec_env) {
  const char *exception;
  uint32_t argv[11];

  float32 data2d[SIZE * SIZE];
  for (int i = 0; i < SIZE * SIZE; i++) {
    data2d[i] = 1.0;
  }
  float32 data1d[SIZE];
  for (int i = 0; i < SIZE; i++) {
    data1d[i] = 1.0;
  }

  InputData arg2_input = initialize_input(module_inst, exec_env,
                                          /*tensor_size=*/SIZE * SIZE,
                                          /*dim=*/2, (uint32_t[]){SIZE, SIZE},
                                          (uint32_t[]){SIZE, 1}, data2d);

  // arg3 to arg10
  InputData arg_inputs[8];
  for (int i = 0; i < 8; i++) {
    arg_inputs[i] = initialize_input(module_inst, exec_env,
                                     /*tensor_size=*/SIZE,
                                     /*dim=*/1, (uint32_t[]){SIZE},
                                     (uint32_t[]){1}, data1d);
  }

  // load main
  wasm_function_inst_t main_func =
      wasm_runtime_lookup_function(module_inst, "_mlir_ciface_main");
  if (!main_func) {
    printk("Fail to find function: main\n");
    return NULL;
  }
  // setup arguments
  float arg0 = 1.2;
  float arg1 = 1.2;
  memcpy(&argv[0], &arg0, sizeof(float));
  memcpy(&argv[1], &arg1, sizeof(float));
  argv[2] = arg2_input.input_ptr;
  for (int i = 0; i < 8; i++) {
    argv[3 + i] = arg_inputs[i].input_ptr;
  }

  // call main
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
  if (!wasm_runtime_call_wasm(exec_env, main_func, 11, argv)) {
    printk("call main fail\n");
  }
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);

  if ((exception = wasm_runtime_get_exception(module_inst)))
    printk("%s\n", exception);

  // read and print output
  for (int i = 0; i < 10; i++) {
    // we read arg7
    printk("%d: %f\n", i, (double)arg_inputs[4].tensor_native_ptr[i]);
  }

  wasm_runtime_destroy_exec_env(exec_env);
  return NULL;
}