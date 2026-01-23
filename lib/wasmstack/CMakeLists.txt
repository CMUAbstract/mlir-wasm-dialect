add_mlir_dialect_library(MLIRWasmStack
        WasmStackDialect.cpp
        WasmStackTypes.cpp
        WasmStackOps.cpp
        WasmStackPasses.cpp
        VerifyWasmStack.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/wasmstack

        DEPENDS
        MLIRWasmStackOpsIncGen
        MLIRWasmStackTypesIncGen
        MLIRWasmStackInterfacesIncGen
        MLIRWasmStackPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        MLIRPass
        MLIRTransforms
        MLIRWasmSSADialect
        MLIRWAMI
        )
