
#include "am_mcu_apollo.h"
#include "bh_assert.h"
#include "bh_log.h"
#include "bh_platform.h"

typedef struct {
  uint32_t base_ptr;
  uint32_t data;
  uint32_t offset;
  uint32_t sizes[2];
  uint32_t strides[2];
} Input2D;

typedef struct {
  uint32_t base_ptr;
  uint32_t data;
  uint32_t offset;
  uint32_t sizes[1];
  uint32_t strides[1];
} Input1D;

typedef struct {
  uint32_t tensor_ptr;
  float32 *tensor_native_ptr;
  uint32_t input_ptr;
  void *input_native_ptr;
} InitializedData;

/*
tensor_size: number of tensor entries (width * height * ...)
*/
InitializedData initialize_input(wasm_module_inst_t module_inst,
                                 wasm_exec_env_t exec_env, uint32_t tensor_size,
                                 uint32_t dim, uint32_t sizes[],
                                 uint32_t strides[], float32 data[]) {
  uint32_t tensor_ptr, input_ptr;
  void *tensor_native_ptr, *input_native_ptr;
  uint32_t argv[1];

  wasm_function_inst_t malloc_fn =
      wasm_runtime_lookup_function(module_inst, "malloc");
  if (!malloc_fn) {
    printk("Fail to find function: malloc\n");
    return (InitializedData){0};
  }

  argv[0] = tensor_size * sizeof(float32);
  wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
  tensor_ptr = argv[0];

  tensor_native_ptr = wasm_runtime_addr_app_to_native(module_inst, tensor_ptr);

  // initialize data
  memcpy((float32 *)tensor_native_ptr, data, tensor_size * sizeof(float32));

  if (dim == 1) {
    argv[0] = sizeof(Input1D);
    wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
    input_ptr = argv[0];
    input_native_ptr = wasm_runtime_addr_app_to_native(module_inst, input_ptr);
    Input1D input = {
        .base_ptr = tensor_ptr,
        .data = tensor_ptr,
        .offset = 0,
        .sizes = {sizes[0]},
        .strides = {strides[0]},
    };
    memcpy((Input1D *)input_native_ptr, &input, sizeof(Input1D));
  } else if (dim == 2) {
    argv[0] = sizeof(Input2D);
    wasm_runtime_call_wasm(exec_env, malloc_fn, 1, argv);
    input_ptr = argv[0];
    input_native_ptr = wasm_runtime_addr_app_to_native(module_inst, input_ptr);
    Input2D input = {
        .base_ptr = tensor_ptr,
        .data = tensor_ptr,
        .offset = 0,
        .sizes = {sizes[0], sizes[1]},
        .strides = {strides[0], strides[1]},
    };
    memcpy((Input2D *)input_native_ptr, &input, sizeof(Input2D));
  }

  InitializedData result = {
      .tensor_ptr = tensor_ptr,
      .tensor_native_ptr = tensor_native_ptr,
      .input_ptr = input_ptr,
      .input_native_ptr = input_native_ptr,
  };

  return result;
}
