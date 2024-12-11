#include "utility.h"

#define SIZE 128

static void *app_instance_main(wasm_module_inst_t module_inst,
                               wasm_exec_env_t exec_env) {
  const char *exception;
  uint32_t argv[3];

  float32 arg0[SIZE * SIZE * SIZE];
  for (int i = 0; i < SIZE * SIZE * SIZE; i++) {
    arg0[i] = 1.0;
  }

  InputData arg0_data =
      initialize_input(module_inst, exec_env,
                       /*tensor size*/ SIZE * SIZE * SIZE,
                       /*dim*/ 3, (uint32_t[]){SIZE, SIZE, SIZE},
                       (uint32_t[]){SIZE * SIZE, SIZE, 1}, arg0);

  float32 arg1[SIZE * SIZE];
  for (int i = 0; i < SIZE * SIZE; i++) {
    arg1[i] = 1.0;
  }
  InputData arg1_data = initialize_input(module_inst, exec_env,
                                         /*tensor size*/ SIZE * SIZE,
                                         /*dim*/ 2, (uint32_t[]){SIZE, SIZE},
                                         (uint32_t[]){SIZE, 1}, arg1);

  float32 arg2[SIZE];
  for (int i = 0; i < SIZE; i++) {
    arg1[i] = 1.0;
  }
  InputData arg2_data =
      initialize_input(module_inst, exec_env,
                       /*tensor size*/ SIZE,
                       /*dim*/ 1, (uint32_t[]){SIZE}, (uint32_t[]){1}, arg2);

  // load main
  wasm_function_inst_t main_func =
      wasm_runtime_lookup_function(module_inst, "_mlir_ciface_main");
  if (!main_func) {
    printk("Fail to find function: main\n");
    return NULL;
  }
  // setup arguments
  argv[0] = arg0_data.input_ptr;
  argv[1] = arg1_data.input_ptr;
  argv[2] = arg2_data.input_ptr;

  // call main
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
  if (!wasm_runtime_call_wasm(exec_env, main_func, 3, argv)) {
    printk("call main fail\n");
  }
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);

  if ((exception = wasm_runtime_get_exception(module_inst)))
    printk("%s\n", exception);

  // read and print output
  for (int i = 0; i < 10; i++) {
    printk("%d: %f\n", i, (double)arg0_data.tensor_native_ptr[i]);
  }

  wasm_runtime_destroy_exec_env(exec_env);
  return NULL;
}