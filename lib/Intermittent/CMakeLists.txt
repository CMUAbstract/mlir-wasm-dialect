add_mlir_dialect_library(MLIRIntermittent
        IntermittentTypes.cpp
        IntermittentDialect.cpp
        IntermittentOps.cpp
        IntermittentPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Wasm


        DEPENDS
        MLIRIntermittentOpsIncGen
        MLIRIntermittentPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
