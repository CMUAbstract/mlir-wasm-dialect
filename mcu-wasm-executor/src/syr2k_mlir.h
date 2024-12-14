#include "utility.h"

#define SIZE 128

static void *app_instance_main(wasm_module_inst_t module_inst,
                               wasm_exec_env_t exec_env) {
  const char *exception;
  uint32_t argv[5];

  float32 data2d[SIZE * SIZE];
  for (int i = 0; i < SIZE * SIZE; i++) {
    data2d[i] = 1.0;
  }

  TensorData arg2_tensor =
      initialize_tensor(module_inst, exec_env,
                        /*tensor_size=*/SIZE * SIZE, data2d);
  TensorData arg3_tensor =
      initialize_tensor(module_inst, exec_env,
                        /*tensor_size=*/SIZE * SIZE, data2d);
  TensorData arg4_tensor =
      initialize_tensor(module_inst, exec_env,
                        /*tensor_size=*/SIZE * SIZE, data2d);

  // load main
  wasm_function_inst_t main_func =
      wasm_runtime_lookup_function(module_inst, "main");
  if (!main_func) {
    printk("Fail to find function: main\n");
    return NULL;
  }
  // setup arguments
  float arg0 = 1.2;
  float arg1 = 1.2;
  memcpy(&argv[0], &arg0, sizeof(float));
  memcpy(&argv[1], &arg1, sizeof(float));
  argv[2] = arg2_tensor.tensor_ptr;
  argv[3] = arg3_tensor.tensor_ptr;
  argv[4] = arg4_tensor.tensor_ptr;

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
    printk("%d: %f\n", i, (double)arg2_tensor.tensor_native_ptr[i]);
  }

  wasm_runtime_destroy_exec_env(exec_env);
  return NULL;
}