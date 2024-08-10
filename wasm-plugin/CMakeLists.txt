add_llvm_library(WasmPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        wasm-plugin.cpp

        DEPENDS
        MLIRWasm
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRWasm
        )
target_include_directories(
        WasmPlugin
        PRIVATE
        "${WASM_BINARY_DIR}/include"
)
