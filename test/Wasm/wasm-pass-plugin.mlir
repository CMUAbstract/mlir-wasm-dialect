// RUN: mlir-opt %s --load-pass-plugin=%wasm_libs/WasmPlugin%shlibext --pass-pipeline="builtin.module(wasm-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
