add_llvm_library(WasmPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        wasm-plugin.cpp

        DEPENDS
        MLIRWasm
        MLIRIntermittent
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRWasm
        MLIRIntermittent
        )
target_include_directories(
        WasmPlugin
        PRIVATE
        "${WASM_BINARY_DIR}/include"
)
