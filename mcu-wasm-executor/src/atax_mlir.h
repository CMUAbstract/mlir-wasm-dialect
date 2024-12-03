#include "am_mcu_apollo.h"
#include "bh_assert.h"
#include "bh_log.h"
#include "bh_platform.h"

static void *app_instance_main(wasm_module_inst_t module_inst,
                               wasm_exec_env_t exec_env) {
  const char *exception;

  uint32 argv[4];

  wasm_function_inst_t malloc_fn =
      wasm_runtime_lookup_function(module_inst, "malloc");
  if (!malloc_fn) {
    printk("Fail to find function: malloc\n");
    return NULL;
  }

  // initialize input A
  argv[0] = 32 * 32 * 4;
  wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
  uint32_t A_ptr = argv[0];
  float32 *A_native_ptr = wasm_runtime_addr_app_to_native(module_inst, A_ptr);

  float32 A[32 * 32];
  for (int i = 0; i < 32 * 32; i++) {
    A[i] = 1.0;
  }
  memcpy(A_native_ptr, A, 32 * 32 * 4);

  // initialize input x
  argv[0] = 32 * 4;
  wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
  uint32_t x_ptr = argv[0];
  float32 *x_native_ptr = wasm_runtime_addr_app_to_native(module_inst, x_ptr);

  float32 x[32 * 4];
  for (int i = 0; i < 32 * 4; i++) {
    x[i] = 1.0;
  }
  memcpy(x_native_ptr, A, 32 * 4);

  // initialize input y
  argv[0] = 32 * 4;
  wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
  uint32_t y_ptr = argv[0];

  // initialize input tmp
  argv[0] = 32 * 4;
  wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
  uint32_t tmp_ptr = argv[0];

  // load main
  wasm_function_inst_t main_func =
      wasm_runtime_lookup_function(module_inst, "main");
  if (!main_func) {
    printk("Fail to find function: main\n");
    return NULL;
  }
  // setup arguments
  argv[0] = A_ptr;
  argv[1] = x_ptr;
  argv[2] = y_ptr;
  argv[3] = tmp_ptr;

  // call main
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);
  if (!wasm_runtime_call_wasm(exec_env, main_func, 4, argv)) {
    printk("call main fail\n");
  }
  am_hal_gpio_state_write(22, AM_HAL_GPIO_OUTPUT_TOGGLE);

  if ((exception = wasm_runtime_get_exception(module_inst)))
    printk("%s\n", exception);

  // read and print output
  float32 *y_native_ptr = wasm_runtime_addr_app_to_native(module_inst, y_ptr);

  for (int i = 0; i < 10; i++) {
    printk("%d: %f\n", i, (double)y_native_ptr[i]);
  }

  wasm_runtime_destroy_exec_env(exec_env);
  return NULL;
}