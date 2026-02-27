llvm_canonicalize_cmake_booleans(MLIR_ENABLE_BINDINGS_PYTHON)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(WASM_TEST_DEPENDS wasm-opt wasm-emit)

# FileCheck, count, not are only CMake targets when building in the LLVM tree.
# When building standalone against an installed LLVM, they exist as binaries in
# the install prefix and don't need to be listed as target dependencies.
if(TARGET FileCheck)
  list(APPEND WASM_TEST_DEPENDS FileCheck count not)
endif()
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND WASM_TEST_DEPENDS WasmPythonModules)
endif()

add_lit_testsuite(check-wasm "Running the wasm regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${WASM_TEST_DEPENDS})
set_target_properties(check-wasm PROPERTIES FOLDER "Tests")

add_lit_testsuites(WASM ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS
                   ${WASM_TEST_DEPENDS})
