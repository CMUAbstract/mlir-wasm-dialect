add_mlir_library(MLIRWasmStackBinaryEmitter
    IndexSpace.cpp
    OpcodeEmitter.cpp
    WasmBinaryEmitter.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/WasmStack

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRWasmStack
    )
