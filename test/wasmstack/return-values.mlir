// RUN: wasm-opt %s --convert-to-wasmstack 2>&1 | FileCheck %s

// Test that stackification pass correctly handles return values.
// This test verifies that return statements properly have their operands
// on the stack before the return instruction.

// CHECK: ConvertToWasmStack pass running on module

module {
  //===--------------------------------------------------------------------===//
  // Test 1: Simple function returning a constant
  //===--------------------------------------------------------------------===//

  // A function that returns a single constant value
  // The constant should be emitted to the stack before the return
  // CHECK-LABEL: wasmstack.func @return_constant
  // CHECK:         wasmstack.i32.const 42
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @return_constant() -> i32 {
    %c = wasmssa.const 42 : i32
    wasmssa.return %c : i32
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Function returning the result of an operation
  //===--------------------------------------------------------------------===//

  // A function that returns the result of an addition
  // The operands should be computed and the result left on stack before return
  // CHECK-LABEL: wasmstack.func @return_add_result
  // CHECK:         wasmstack.i32.const 10
  // CHECK-NEXT:    wasmstack.i32.const 20
  // CHECK-NEXT:    wasmstack.add : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @return_add_result() -> i32 {
    %a = wasmssa.const 10 : i32
    %b = wasmssa.const 20 : i32
    %sum = wasmssa.add %a %b : i32
    wasmssa.return %sum : i32
  }

  //===--------------------------------------------------------------------===//
  // Test 3: Function returning multiple values
  //===--------------------------------------------------------------------===//

  // A function that returns two values
  // Both values should be emitted to the stack before return
  // CHECK-LABEL: wasmstack.func @return_multiple
  // CHECK:         wasmstack.i32.const 1
  // CHECK-NEXT:    wasmstack.i32.const 2
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @return_multiple() -> (i32, i32) {
    %a = wasmssa.const 1 : i32
    %b = wasmssa.const 2 : i32
    wasmssa.return %a, %b : i32, i32
  }

  //===--------------------------------------------------------------------===//
  // Test 4: Function returning multiple computed values
  //===--------------------------------------------------------------------===//

  // A function that computes and returns multiple values
  // Multi-result returns may materialize intermediate values to locals.
  // CHECK-LABEL: wasmstack.func @return_multiple_computed
  // CHECK:         wasmstack.i32.const 10
  // CHECK:         wasmstack.i32.const 20
  // CHECK:         wasmstack.add : i32
  // CHECK:         wasmstack.local.set
  // CHECK:         wasmstack.mul : i32
  // CHECK:         wasmstack.local.set
  // CHECK:         wasmstack.local.get
  // CHECK:         wasmstack.local.get
  // CHECK:         wasmstack.return
  wasmssa.func @return_multiple_computed() -> (i32, i32) {
    %a = wasmssa.const 10 : i32
    %b = wasmssa.const 20 : i32
    %sum = wasmssa.add %a %b : i32
    %prod = wasmssa.mul %a %b : i32
    wasmssa.return %sum, %prod : i32, i32
  }

  //===--------------------------------------------------------------------===//
  // Test 5: Function returning different types
  //===--------------------------------------------------------------------===//

  // A function that returns different numeric types
  // CHECK-LABEL: wasmstack.func @return_mixed_types
  // CHECK:         wasmstack.i32.const 42
  // CHECK-NEXT:    wasmstack.i64.const 100
  // CHECK-NEXT:    wasmstack.f32.const 3.140000e+00
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @return_mixed_types() -> (i32, i64, f32) {
    %i = wasmssa.const 42 : i32
    %l = wasmssa.const 100 : i64
    %f = wasmssa.const 3.14 : f32
    wasmssa.return %i, %l, %f : i32, i64, f32
  }

  //===--------------------------------------------------------------------===//
  // Test 6: Function with void return (no operands)
  //===--------------------------------------------------------------------===//

  // A function that returns nothing - baseline to ensure bare return works
  // This should pass as it has no operands
  // CHECK-LABEL: wasmstack.func @return_void
  // CHECK:         wasmstack.return
  wasmssa.func @return_void() {
    wasmssa.return
  }

  //===--------------------------------------------------------------------===//
  // Test 7: Reusing value for return
  //===--------------------------------------------------------------------===//

  // A function that uses a value multiple times before returning one
  // This tests interaction between multi-use values and return
  // CHECK-LABEL: wasmstack.func @return_with_reuse
  // CHECK:         wasmstack.local 0 : i32
  // CHECK:         wasmstack.i32.const 5
  // CHECK-NEXT:    wasmstack.i32.const 3
  // CHECK-NEXT:    wasmstack.add : i32
  // CHECK-NEXT:    wasmstack.local.tee 0 : i32
  // CHECK-NEXT:    wasmstack.local.get 0 : i32
  // CHECK-NEXT:    wasmstack.mul : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @return_with_reuse() -> i32 {
    %a = wasmssa.const 5 : i32
    %b = wasmssa.const 3 : i32
    %sum = wasmssa.add %a %b : i32
    // Use sum twice, then return the result
    %result = wasmssa.mul %sum %sum : i32
    wasmssa.return %result : i32
  }
}
