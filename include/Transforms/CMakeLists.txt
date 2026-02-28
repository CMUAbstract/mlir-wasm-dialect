set(LLVM_TARGET_DEFINITIONS TransformsPasses.td)
mlir_tablegen(TransformsPasses.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(MLIRTransformsPassesIncGen)
add_dependencies(mlir-headers MLIRTransformsPassesIncGen)
