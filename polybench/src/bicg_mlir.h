#include "utility.h"

#define SIZE 256

static void *app_instance_main(wasm_module_inst_t module_inst,
                               wasm_exec_env_t exec_env) {
  const char *exception;
  uint32_t argv[5];

  // initialize input A
  float32 A[SIZE * SIZE];
  for (int i = 0; i < SIZE * SIZE; i++) {
    A[i] = 1.0;
  }

  TensorData A_data = initialize_tensor(module_inst, exec_env, SIZE * SIZE, A);

  // initialize input s
  float32 s[SIZE];
  for (int i = 0; i < SIZE; i++) {
    s[i] = 1.0;
  }
  TensorData s_data = initialize_tensor(module_inst, exec_env, SIZE, s);
  TensorData q_data = initialize_tensor(module_inst, exec_env, SIZE, s);
  TensorData p_data = initialize_tensor(module_inst, exec_env, SIZE, s);
  TensorData r_data = initialize_tensor(module_inst, exec_env, SIZE, s);

  // load main
  wasm_function_inst_t main_func =
      wasm_runtime_lookup_function(module_inst, "main");
  if (!main_func) {
    printk("Fail to find function: main\n");
    return NULL;
  }
  // setup arguments
  argv[0] = A_data.tensor_ptr;
  argv[1] = s_data.tensor_ptr;
  argv[2] = q_data.tensor_ptr;
  argv[3] = p_data.tensor_ptr;
  argv[4] = r_data.tensor_ptr;

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