include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_wasm`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_wasm.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(WasmPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT WasmPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_wasm"
  TD_FILE dialects/WasmOps.td
  SOURCES
    dialects/wasm.py
  DIALECT_NAME wasm)

declare_mlir_python_extension(WasmPythonSources.Extension
  MODULE_NAME _wasmDialects
  ADD_TO_PARENT WasmPythonSources
  SOURCES
    WasmExtension.cpp
  EMBED_CAPI_LINK_LIBS
    WasmCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(WasmPythonCAPI
  INSTALL_COMPONENT WasmPythonModules
  INSTALL_DESTINATION python_packages/wasm/mlir_wasm/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/wasm/mlir_wasm/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    WasmPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(WasmPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/wasm/mlir_wasm"
  INSTALL_PREFIX "python_packages/wasm/mlir_wasm"
  DECLARED_SOURCES
    WasmPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    WasmPythonCAPI
  )
