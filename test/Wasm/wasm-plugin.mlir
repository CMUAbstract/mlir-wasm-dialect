// RUN: mlir-opt %s --load-dialect-plugin=%wasm_libs/WasmPlugin%shlibext --pass-pipeline="builtin.module(wasm-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @wasm_types(%arg0: !wasm.custom<"10">)
  func.func @wasm_types(%arg0: !wasm.custom<"10">) {
    return
  }
}
