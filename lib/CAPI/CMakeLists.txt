add_mlir_public_c_api_library(WasmCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRWasm
)
