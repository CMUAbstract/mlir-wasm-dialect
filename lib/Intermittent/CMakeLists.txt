add_mlir_dialect_library(MLIRIntermittent
        IntermittentTypes.cpp
        IntermittentDialect.cpp
        IntermittentOps.cpp
        Passes/PolygeistToIntermittent.cpp
        Passes/PreprocessIntermittentToWasm.cpp
        Passes/ConvertIntermittentToWasm.cpp
        Passes/PreprocessIntermittentToStd.cpp
        Passes/ConvertNonVolatileToMemRefForIntermittentToStd.cpp
        Passes/ConvertIntermittentToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Wasm
        ${PROJECT_SOURCE_DIR}/include/Intermittent


        DEPENDS
        MLIRIntermittentOpsIncGen
        MLIRIntermittentPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRPass
        MLIRSupport
        MLIRTransforms
        )
