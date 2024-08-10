# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own wasm dialect.
add_mlir_aggregate(WasmCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  WasmCAPI
)

add_llvm_executable(wasm-capi-test
  wasm-capi-test.c
)
llvm_update_compile_flags(wasm-capi-test)
target_link_libraries(wasm-capi-test
  PRIVATE WasmCAPITestLib)
