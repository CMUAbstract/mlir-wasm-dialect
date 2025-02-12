
set(LLVM_TARGET_DEFINITIONS DContOps.td)
mlir_tablegen(DContOps.h.inc -gen-op-decls)
mlir_tablegen(DContOps.cpp.inc -gen-op-defs)
mlir_tablegen(DContTypes.h.inc -gen-typedef-decls -typedefs-dialect=dcont)
mlir_tablegen(DContTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=dcont)
mlir_tablegen(DContDialect.h.inc -gen-dialect-decls -dialect=dcont)
mlir_tablegen(DContDialect.cpp.inc -gen-dialect-defs -dialect=dcont)
add_public_tablegen_target(MLIRDContOpsIncGen)
add_dependencies(mlir-headers MLIRDContOpsIncGen)

add_mlir_doc(DContDialect DContDialect DCont/ -gen-dialect-doc)
add_mlir_doc(DContOps DContOps DCont/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS DContPasses.td)
mlir_tablegen(DContPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRDContPassesIncGen)
