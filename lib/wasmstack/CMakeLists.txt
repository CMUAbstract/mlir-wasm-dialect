add_mlir_dialect_library(MLIRWasmStack
        WasmStackDialect.cpp
        WasmStackTypes.cpp
        WasmStackOps.cpp
        WasmStackPasses.cpp
        StackificationAnalysis.cpp
        LocalAllocator.cpp
        TreeWalker.cpp
        WasmStackEmitter.cpp
        VerifyWasmStack.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/wasmstack

        DEPENDS
        MLIRWasmStackAttrsIncGen
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
