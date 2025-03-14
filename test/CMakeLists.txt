llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(WASM_TEST_DEPENDS
        FileCheck count not
        wasm-capi-test
        wasm-opt
        wasm-translate
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND WASM_TEST_DEPENDS WasmPythonModules)
endif()

add_lit_testsuite(check-wasm "Running the wasm regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${WASM_TEST_DEPENDS}
        )
set_target_properties(check-wasm PROPERTIES FOLDER "Tests")

add_lit_testsuites(WASM ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${WASM_TEST_DEPENDS})

add_subdirectory(CAPI)
