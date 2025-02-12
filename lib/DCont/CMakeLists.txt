add_mlir_dialect_library(MLIRDCont
        DContTypes.cpp
        DContDialect.cpp
        DContOps.cpp
        DContPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/DCont


        DEPENDS
        MLIRDContOpsIncGen
        MLIRDContPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
