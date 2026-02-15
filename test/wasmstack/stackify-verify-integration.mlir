// RUN: wasm-opt %s --convert-to-wasmstack --verify-wasmstack 2>&1 | FileCheck %s

// Integration tests: Run stackification followed by verification
// These tests ensure that the stackification pass produces valid WasmStack IR
// that passes the verification pass.
//
// Note: Functions with parameters require special !wasm<local T> type wrappers
// in WasmSSA, so we only test parameterless functions here. For functions with
// parameters, see full-pipeline-verify.mlir which starts from standard MLIR.


module {

  //===----------------------------------------------------------------------===//
  // Basic Arithmetic Patterns
  //===----------------------------------------------------------------------===//

  // Simple constant addition
  // CHECK-LABEL: wasmstack.func @const_add
  wasmssa.func @const_add() -> i32 {
    %a = wasmssa.const 10 : i32
    %b = wasmssa.const 20 : i32
    %c = wasmssa.add %a %b : i32
    wasmssa.return %c : i32
  }

  // Chain of arithmetic
  // CHECK-LABEL: wasmstack.func @arith_chain
  wasmssa.func @arith_chain() -> i32 {
    %a = wasmssa.const 10 : i32
    %b = wasmssa.const 5 : i32
    %c = wasmssa.const 2 : i32
    %sum = wasmssa.add %a %b : i32
    %prod = wasmssa.mul %sum %c : i32
    wasmssa.return %prod : i32
  }

  // Multi-use constant (rematerialization)
  // CHECK-LABEL: wasmstack.func @remat_const
  wasmssa.func @remat_const() -> i32 {
    %x = wasmssa.const 5 : i32
    %doubled = wasmssa.add %x %x : i32
    wasmssa.return %doubled : i32
  }

  // Multi-use non-constant (tee pattern)
  // CHECK-LABEL: wasmstack.func @multi_use_tee
  wasmssa.func @multi_use_tee() -> i32 {
    %a = wasmssa.const 10 : i32
    %b = wasmssa.const 20 : i32
    %sum = wasmssa.add %a %b : i32
    %result = wasmssa.mul %sum %sum : i32
    wasmssa.return %result : i32
  }

  // Deep expression tree
  // CHECK-LABEL: wasmstack.func @deep_expr
  wasmssa.func @deep_expr() -> i32 {
    %a = wasmssa.const 1 : i32
    %b = wasmssa.const 2 : i32
    %c = wasmssa.const 3 : i32
    %d = wasmssa.const 4 : i32
    %ab = wasmssa.add %a %b : i32
    %cd = wasmssa.add %c %d : i32
    %result = wasmssa.mul %ab %cd : i32
    wasmssa.return %result : i32
  }

  // Subtraction and division
  // CHECK-LABEL: wasmstack.func @sub_div
  wasmssa.func @sub_div() -> i32 {
    %a = wasmssa.const 100 : i32
    %b = wasmssa.const 7 : i32
    %sub = wasmssa.sub %a %b : i32
    %div = wasmssa.div_si %sub %b : i32
    wasmssa.return %div : i32
  }

  // Bitwise operations
  // CHECK-LABEL: wasmstack.func @bitwise_ops
  wasmssa.func @bitwise_ops() -> i32 {
    %a = wasmssa.const 0xFF : i32
    %b = wasmssa.const 0x0F : i32
    %and_res = wasmssa.and %a %b : i32
    %or_res = wasmssa.or %and_res %b : i32
    %xor_res = wasmssa.xor %or_res %b : i32
    wasmssa.return %xor_res : i32
  }

  // Shift operations
  // CHECK-LABEL: wasmstack.func @shift_ops
  wasmssa.func @shift_ops() -> i32 {
    %a = wasmssa.const 1 : i32
    %shift = wasmssa.const 4 : i32
    %shl = wasmssa.shl %a by %shift bits : i32
    wasmssa.return %shl : i32
  }

  //===----------------------------------------------------------------------===//
  // Comparison Operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: wasmstack.func @comparisons
  wasmssa.func @comparisons() -> i32 {
    %a = wasmssa.const 10 : i32
    %b = wasmssa.const 5 : i32
    %lt = wasmssa.lt_si %a %b : i32 -> i32
    wasmssa.return %lt : i32
  }

  // CHECK-LABEL: wasmstack.func @eqz_test
  wasmssa.func @eqz_test() -> i32 {
    %a = wasmssa.const 0 : i32
    %result = wasmssa.eqz %a : i32 -> i32
    wasmssa.return %result : i32
  }

  //===----------------------------------------------------------------------===//
  // Float Operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: wasmstack.func @f32_arith
  wasmssa.func @f32_arith() -> f32 {
    %a = wasmssa.const 3.14 : f32
    %b = wasmssa.const 2.0 : f32
    %sum = wasmssa.add %a %b : f32
    %prod = wasmssa.mul %sum %b : f32
    wasmssa.return %prod : f32
  }

  // CHECK-LABEL: wasmstack.func @f64_arith
  wasmssa.func @f64_arith() -> f64 {
    %a = wasmssa.const 3.14159265358979 : f64
    %b = wasmssa.const 2.71828182845904 : f64
    %sum = wasmssa.add %a %b : f64
    wasmssa.return %sum : f64
  }

  // CHECK-LABEL: wasmstack.func @f32_compare
  wasmssa.func @f32_compare() -> i32 {
    %a = wasmssa.const 1.5 : f32
    %b = wasmssa.const 2.5 : f32
    %result = wasmssa.lt %a %b : f32 -> i32
    wasmssa.return %result : i32
  }

  //===----------------------------------------------------------------------===//
  // 64-bit Integer Operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: wasmstack.func @i64_arith
  wasmssa.func @i64_arith() -> i64 {
    %a = wasmssa.const 1000000000000 : i64
    %b = wasmssa.const 2000000000000 : i64
    %sum = wasmssa.add %a %b : i64
    %c = wasmssa.const 2 : i64
    %result = wasmssa.mul %sum %c : i64
    wasmssa.return %result : i64
  }

  //===----------------------------------------------------------------------===//
  // Void Functions
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: wasmstack.func @void_func
  wasmssa.func @void_func() {
    wasmssa.return
  }

}
