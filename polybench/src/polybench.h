static void *app_instance_main(wasm_module_inst_t module_inst,
                               wasm_exec_env_t exec_env) {
  const char *exception;
  uint32_t argv[2];

  // load main
  wasm_function_inst_t main_func =
      wasm_runtime_lookup_function(module_inst, "main");
  if (!main_func) {
    printf("Fail to find function: main\n");
    return NULL;
  }

  // call main
  // MLIR-produced Wasm function does not have any arguments, but it is safe to
  // pass dummy arguments. LLVM-produced Wasm function has two arguments, but
  // they are ignored anyway.
  if (!wasm_runtime_call_wasm(exec_env, main_func, 2, argv)) {
    printf("call main fail\n");
  }

  if ((exception = wasm_runtime_get_exception(module_inst)))
    printf("%s\n", exception);

  wasm_runtime_destroy_exec_env(exec_env);
  return NULL;
}