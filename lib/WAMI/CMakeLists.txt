add_mlir_dialect_library(MLIRWAMI
  WAMIDialect.cpp
  WAMITypes.cpp
  WAMIOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/WAMI

  DEPENDS
  MLIRWAMIOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRWasmSSADialect
)
