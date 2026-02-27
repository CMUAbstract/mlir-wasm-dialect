set(LLVM_TARGET_DEFINITIONS CoroPasses.td)
mlir_tablegen(CoroPasses.h.inc -gen-pass-decls -name Coro)
add_public_tablegen_target(MLIRCoroPassesIncGen)
add_dependencies(mlir-headers MLIRCoroPassesIncGen)
