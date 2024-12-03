#include "am_mcu_apollo.h"
#include "bh_assert.h"
#include "bh_log.h"
#include "bh_platform.h"
#include "input_data.h"

#define INPUT_TENSOR_SIZE 3136
#define OUTPUT_TENSOR_SIZE 40

static void *app_instance_main(wasm_module_inst_t module_inst,
                               wasm_exec_env_t exec_env) {

  const char *exception;

  float32 *input_tensor_native_ptr = NULL;
  float32 *output_tensor_native_ptr = NULL;
  uint32 argv[2];

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
    printk("%d: %lf\n", i, (double)output_tensor_native_ptr[i]);
  }

  wasm_runtime_destroy_exec_env(exec_env);
  return NULL;
}