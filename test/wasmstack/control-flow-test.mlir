// RUN: wasm-opt %s --convert-to-wasmstack 2>&1 | FileCheck %s

// Test control flow and comparison operations

// CHECK: ConvertToWasmStack pass running on module

module {
  // Test comparison operation (operands pushed left-to-right for stack evaluation)
  // %a (lhs) is pushed first (bottom), then %b (rhs) (top), so lt_s compares bottom < top
  // CHECK-LABEL: wasmstack.func @test_compare
  // CHECK:         wasmstack.i32.const 10
  // CHECK-NEXT:    wasmstack.i32.const 5
  // CHECK-NEXT:    wasmstack.lt_s : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @test_compare() -> i32 {
    %a = wasmssa.const 10 : i32
    %b = wasmssa.const 5 : i32
    %cmp = wasmssa.lt_si %a %b : i32 -> i32
    wasmssa.return %cmp : i32
  }

  // Test equality comparison
  // CHECK-LABEL: wasmstack.func @test_eq
  // CHECK:         wasmstack.i32.const 42
  // CHECK-NEXT:    wasmstack.i32.const 42
  // CHECK-NEXT:    wasmstack.eq : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @test_eq() -> i32 {
    %a = wasmssa.const 42 : i32
    %b = wasmssa.const 42 : i32
    %result = wasmssa.eq %a %b : i32 -> i32
    wasmssa.return %result : i32
  }

  // Test division (lhs pushed first, rhs pushed second)
  // CHECK-LABEL: wasmstack.func @test_div
  // CHECK:         wasmstack.i32.const 100
  // CHECK-NEXT:    wasmstack.i32.const 10
  // CHECK-NEXT:    wasmstack.div_s : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @test_div() -> i32 {
    %a = wasmssa.const 100 : i32
    %b = wasmssa.const 10 : i32
    %result = wasmssa.div_si %a %b : i32
    wasmssa.return %result : i32
  }

  // Test remainder
  // CHECK-LABEL: wasmstack.func @test_rem
  // CHECK:         wasmstack.i32.const 17
  // CHECK-NEXT:    wasmstack.i32.const 5
  // CHECK-NEXT:    wasmstack.rem_s : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @test_rem() -> i32 {
    %a = wasmssa.const 17 : i32
    %b = wasmssa.const 5 : i32
    %result = wasmssa.rem_si %a %b : i32
    wasmssa.return %result : i32
  }

  // Test bitwise AND
  // CHECK-LABEL: wasmstack.func @test_and
  // CHECK:         wasmstack.i32.const 15
  // CHECK-NEXT:    wasmstack.i32.const 7
  // CHECK-NEXT:    wasmstack.and : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @test_and() -> i32 {
    %a = wasmssa.const 15 : i32
    %b = wasmssa.const 7 : i32
    %result = wasmssa.and %a %b : i32
    wasmssa.return %result : i32
  }

  // Test bitwise OR
  // CHECK-LABEL: wasmstack.func @test_or
  // CHECK:         wasmstack.i32.const 8
  // CHECK-NEXT:    wasmstack.i32.const 3
  // CHECK-NEXT:    wasmstack.or : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @test_or() -> i32 {
    %a = wasmssa.const 8 : i32
    %b = wasmssa.const 3 : i32
    %result = wasmssa.or %a %b : i32
    wasmssa.return %result : i32
  }

  // Test shift left
  // CHECK-LABEL: wasmstack.func @test_shl
  // CHECK:         wasmstack.i32.const 1
  // CHECK-NEXT:    wasmstack.i32.const 4
  // CHECK-NEXT:    wasmstack.shl : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @test_shl() -> i32 {
    %a = wasmssa.const 1 : i32
    %b = wasmssa.const 4 : i32
    %result = wasmssa.shl %a by %b bits : i32
    wasmssa.return %result : i32
  }

  // Test eqz (equal to zero)
  // CHECK-LABEL: wasmstack.func @test_eqz
  // CHECK:         wasmstack.i32.const 0
  // CHECK-NEXT:    wasmstack.eqz : i32
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @test_eqz() -> i32 {
    %a = wasmssa.const 0 : i32
    %result = wasmssa.eqz %a : i32 -> i32
    wasmssa.return %result : i32
  }

  // Test float comparisons
  // CHECK-LABEL: wasmstack.func @test_float_cmp
  // CHECK:         wasmstack.f64.const 3.14
  // CHECK-NEXT:    wasmstack.f64.const 2.71
  // CHECK-NEXT:    wasmstack.lt : f64
  // CHECK-NEXT:    wasmstack.return
  wasmssa.func @test_float_cmp() -> i32 {
    %a = wasmssa.const 3.14 : f64
    %b = wasmssa.const 2.71 : f64
    %result = wasmssa.lt %a %b : f64 -> i32
    wasmssa.return %result : i32
  }
}
