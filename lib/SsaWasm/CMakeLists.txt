file(GLOB_RECURSE CONVERSION_SRC_FILES ConversionPatterns/*.cpp)

add_mlir_dialect_library(MLIRSsaWasm
        SsaWasmTypes.cpp
        SsaWasmDialect.cpp
        SsaWasmOps.cpp
        SsaWasmPasses.cpp
        ${CONVERSION_SRC_FILES}

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/SsaWasm


        DEPENDS
        MLIRSsaWasmOpsIncGen
        MLIRSsaWasmPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
