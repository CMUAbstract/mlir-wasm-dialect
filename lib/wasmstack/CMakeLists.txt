add_mlir_dialect_library(MLIRWasmStack
        WasmStackDialect.cpp
        WasmStackTypes.cpp
        WasmStackOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/wasmstack

        DEPENDS
        MLIRWasmStackOpsIncGen
        MLIRWasmStackTypesIncGen
        MLIRWasmStackInterfacesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        )
