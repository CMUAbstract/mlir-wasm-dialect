get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRRegisterAllPasses
        MLIRWAMI
        MLIRWasmStack
        )
add_llvm_executable(wasm-opt wasm-opt.cpp)

llvm_update_compile_flags(wasm-opt)
target_link_libraries(wasm-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(wasm-opt)
