# Generate dialect declarations
set(LLVM_TARGET_DEFINITIONS WasmStackOps.td)
mlir_tablegen(WasmStackOps.h.inc -gen-op-decls)
mlir_tablegen(WasmStackOps.cpp.inc -gen-op-defs)
mlir_tablegen(WasmStackDialect.h.inc -gen-dialect-decls -dialect=wasmstack)
mlir_tablegen(WasmStackDialect.cpp.inc -gen-dialect-defs -dialect=wasmstack)
add_public_tablegen_target(MLIRWasmStackOpsIncGen)
add_dependencies(mlir-headers MLIRWasmStackOpsIncGen)

# Generate type declarations
set(LLVM_TARGET_DEFINITIONS WasmStackTypes.td)
mlir_tablegen(WasmStackTypes.h.inc -gen-typedef-decls -typedefs-dialect=wasmstack)
mlir_tablegen(WasmStackTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=wasmstack)
add_public_tablegen_target(MLIRWasmStackTypesIncGen)
add_dependencies(mlir-headers MLIRWasmStackTypesIncGen)

# Generate interface declarations
set(LLVM_TARGET_DEFINITIONS WasmStackInterfaces.td)
mlir_tablegen(WasmStackInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(WasmStackInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRWasmStackInterfacesIncGen)
add_dependencies(mlir-headers MLIRWasmStackInterfacesIncGen)

# Documentation
add_mlir_doc(WasmStackDialect WasmStackDialect wasmstack/ -gen-dialect-doc)
add_mlir_doc(WasmStackOps WasmStackOps wasmstack/ -gen-op-doc)
