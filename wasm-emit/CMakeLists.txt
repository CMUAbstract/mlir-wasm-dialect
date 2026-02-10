set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(wasm-emit
  wasm-emit.cpp
  )
llvm_update_compile_flags(wasm-emit)
target_link_libraries(wasm-emit
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  MLIRWasmStack
  MLIRWasmStackBinaryEmitter
  )

mlir_check_link_libraries(wasm-emit)
