#include "am_mcu_apollo.h"
#include "bh_assert.h"
#include "bh_log.h"
#include "bh_platform.h"
#include "input_data.h"

typedef struct {
  uint32_t base_ptr;
  uint32_t data;
  uint32_t offset;
  uint32_t sizes[3];
  uint32_t strides[3];
} Input;

typedef struct {
  uint32_t base_ptr;
  uint32_t data;
  uint32_t offset;
  uint32_t sizes[2];
  uint32_t strides[2];
} Output;

static void *app_instance_main(wasm_module_inst_t module_inst,
                               wasm_exec_env_t exec_env) {
  const char *exception;

  Input *input_native_ptr = NULL;
  float32 *input_tensor_native_ptr = NULL;
  Output *output_native_ptr = NULL;
  float32 *output_tensor_native_ptr = NULL;
  uint32 argv[2];

  wasm_function_inst_t malloc_fn =
      wasm_runtime_lookup_function(module_inst, "malloc");
  if (!malloc_fn) {
    printk("Fail to find function: malloc\n");
    return NULL;
  }

  argv[0] = 32 * 32 * 4;
  wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
  uint32_t A_ptr = argv[0];

  input_tensor_native_ptr = wasm_runtime_addr_app_to_native(module_inst, A_ptr);

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
  output_native_ptr = wasm_runtime_addr_app_to_native(module_inst, output_ptr);

  Output output;
  memcpy(&output, output_native_ptr, sizeof(Output));
  output_tensor_native_ptr =
      wasm_runtime_addr_app_to_native(module_inst, output.data);

  for (int i = 0; i < 10; i++) {
    printk("%d: %f\n", i, (double)output_tensor_native_ptr[i]);
  }

  wasm_runtime_destroy_exec_env(exec_env);
  return NULL;
}