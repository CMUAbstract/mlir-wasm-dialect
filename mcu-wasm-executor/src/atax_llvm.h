#include "utility.h"

#define SIZE 256

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
  uint32_t argv[4];

  float32 A[SIZE * SIZE];
  for (int i = 0; i < SIZE * SIZE; i++) {
    A[i] = 1.0;
  }

  InitializedData A_data = initialize_input(module_inst, exec_env,
                                            /*tensor size*/ SIZE * SIZE,
                                            /*dim*/ 2, (uint32_t[]){SIZE, SIZE},
                                            (uint32_t[]){SIZE, 1}, A);

  float32 x[SIZE * 4];
  for (int i = 0; i < SIZE * 4; i++) {
    x[i] = 1.0;
  }
  // initialize input x
  InitializedData x_data = initialize_input(
      module_inst, exec_env, SIZE, 1, (uint32_t[]){SIZE}, (uint32_t[]){1}, x);

  // initialize input y
  InitializedData y_data = initialize_input(
      module_inst, exec_env, SIZE, 1, (uint32_t[]){SIZE}, (uint32_t[]){1}, x);

  // initialize input tmp
  InitializedData tmp_data = initialize_input(
      module_inst, exec_env, SIZE, 1, (uint32_t[]){SIZE}, (uint32_t[]){1}, x);

  // load main
  wasm_function_inst_t main_func =
      wasm_runtime_lookup_function(module_inst, "_mlir_ciface_main");
  if (!main_func) {
    printk("Fail to find function: main\n");
    return NULL;
  }
  // setup arguments
  argv[0] = A_data.input_ptr;
  argv[1] = x_data.input_ptr;
  argv[2] = y_data.input_ptr;
  argv[3] = tmp_data.input_ptr;

  // call main
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
  if (!wasm_runtime_call_wasm(exec_env, main_func, 4, argv)) {
    printk("call main fail\n");
  }
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);

  if ((exception = wasm_runtime_get_exception(module_inst)))
    printk("%s\n", exception);

  // read and print output
  float32 *y_native_ptr = y_data.tensor_native_ptr;

  for (int i = 0; i < 10; i++) {
    printk("%d: %f\n", i, (double)y_native_ptr[i]);
  }

  wasm_runtime_destroy_exec_env(exec_env);
  return NULL;
}