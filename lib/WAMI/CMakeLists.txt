add_mlir_dialect_library(MLIRWAMI
  WAMIDialect.cpp
  WAMITypes.cpp
  WAMIOps.cpp
  WAMIPasses.cpp
  ConversionPatterns/WAMIConversionUtils.cpp
  ConversionPatterns/WAMIConvertArith.cpp
  ConversionPatterns/WAMIConvertMath.cpp
  ConversionPatterns/WAMIConvertFunc.cpp
  ConversionPatterns/WAMIConvertScf.cpp
  ConversionPatterns/WAMIConvertMemref.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/WAMI

  DEPENDS
  MLIRWAMIOpsIncGen
  MLIRWAMIPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
  MLIRWasmSSADialect
)
