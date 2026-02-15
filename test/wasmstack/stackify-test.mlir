// RUN: wasm-opt %s --convert-to-wasmstack 2>&1 | FileCheck %s

// Test basic stackification on WasmSSA dialect

// Then MLIR output on stdout
// CHECK-LABEL: wasmstack.func @const_add
// CHECK:         wasmstack.i32.const 10
// CHECK-NEXT:    wasmstack.i32.const 20
// CHECK-NEXT:    wasmstack.add : i32
// CHECK-NEXT:    wasmstack.return

module {
  // Simple function with constants - tests single-use movement
  wasmssa.func @const_add() -> i32 {
    %a = wasmssa.const 10 : i32
    %b = wasmssa.const 20 : i32
    %c = wasmssa.add %a %b : i32
    wasmssa.return %c : i32
  }

  // Multi-use constant - original is used for first operand, cloned for second
  // Constants are cheap to rematerialize, so no local variable needed
  // CHECK-LABEL: wasmstack.func @multi_use_const
  // CHECK:         wasmstack.i32.const 5
  // CHECK-NEXT:    wasmstack.i32.const 5
  // CHECK-NEXT:    wasmstack.add : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @multi_use_const() -> i32 {
    %x = wasmssa.const 5 : i32
    %doubled = wasmssa.add %x %x : i32
    wasmssa.return %doubled : i32
  }

  // Multi-use of non-rematerializable value (add result) - should use tee
  // One use can consume from stack, other uses will use local.get
  // CHECK-LABEL: wasmstack.func @multi_use_add
  // CHECK:         wasmstack.local 0 : i32
  // CHECK:         wasmstack.local.tee 0 : i32
  // CHECK-NEXT:    wasmstack.local.get 0 : i32
  // CHECK-NEXT:    wasmstack.add : i32
  wasmssa.func @multi_use_add() -> i32 {
    %a = wasmssa.const 10 : i32
    %b = wasmssa.const 20 : i32
    %sum = wasmssa.add %a %b : i32
    // Using sum twice - add is not rematerializable, so needs tee
    %doubled = wasmssa.add %sum %sum : i32
    wasmssa.return %doubled : i32
  }
}
