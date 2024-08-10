add_mlir_dialect(WasmOps wasm)
add_mlir_doc(WasmDialect WasmDialect Wasm/ -gen-dialect-doc)
add_mlir_doc(WasmOps WasmOps Wasm/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS WasmPasses.td)
mlir_tablegen(WasmPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRWasmPassesIncGen)
