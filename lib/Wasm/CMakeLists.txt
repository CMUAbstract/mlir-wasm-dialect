file(GLOB_RECURSE CONVERSION_SRC_FILES ConversionPatterns/*.cpp)

add_mlir_dialect_library(MLIRWasm
        WasmTypes.cpp
        WasmDialect.cpp
        WasmOps.cpp
        WasmPasses.cpp
        VariableAnalysis.cpp
        ${CONVERSION_SRC_FILES}

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Wasm


        DEPENDS
        MLIRWasmOpsIncGen
        MLIRWasmPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
