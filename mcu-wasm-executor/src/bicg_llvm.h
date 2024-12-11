#include "utility.h"

#define SIZE 256

static void *app_instance_main(wasm_module_inst_t module_inst,
                               wasm_exec_env_t exec_env) {
  const char *exception;
  uint32_t argv[5];

  float32 A[SIZE * SIZE];
  for (int i = 0; i < SIZE * SIZE; i++) {
    A[i] = 1.0;
  }

  InputData A_data = initialize_input(module_inst, exec_env,
                                      /*tensor size*/ SIZE * SIZE,
                                      /*dim*/ 2, (uint32_t[]){SIZE, SIZE},
                                      (uint32_t[]){SIZE, 1}, A);

  float32 s[SIZE * 4];
  for (int i = 0; i < SIZE * 4; i++) {
    s[i] = 1.0;
  }
  InputData s_data = initialize_input(module_inst, exec_env, SIZE, 1,
                                      (uint32_t[]){SIZE}, (uint32_t[]){1}, s);
  InputData q_data = initialize_input(module_inst, exec_env, SIZE, 1,
                                      (uint32_t[]){SIZE}, (uint32_t[]){1}, s);
  InputData p_data = initialize_input(module_inst, exec_env, SIZE, 1,
                                      (uint32_t[]){SIZE}, (uint32_t[]){1}, s);
  InputData r_data = initialize_input(module_inst, exec_env, SIZE, 1,
                                      (uint32_t[]){SIZE}, (uint32_t[]){1}, s);

  // load main
  wasm_function_inst_t main_func =
      wasm_runtime_lookup_function(module_inst, "_mlir_ciface_main");
  if (!main_func) {
    printk("Fail to find function: main\n");
    return NULL;
  }
  // setup arguments
  argv[0] = s_data.input_ptr;
  argv[1] = q_data.input_ptr;
  argv[2] = p_data.input_ptr;
  argv[3] = r_data.input_ptr;

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
    printk("%d: %f\n", i, (double)s_data.tensor_native_ptr[i]);
  }
  for (int i = 0; i < 10; i++) {
    printk("%d: %f\n", i, (double)q_data.tensor_native_ptr[i]);
  }

  wasm_runtime_destroy_exec_env(exec_env);
  return NULL;
}