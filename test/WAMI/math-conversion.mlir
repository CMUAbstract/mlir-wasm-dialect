// RUN: wasm-opt %s --wami-convert-math --wami-convert-arith --wami-convert-func --reconcile-unrealized-casts | FileCheck %s
// RUN: wasm-opt %s --wami-convert-math --wami-convert-arith --wami-convert-func --reconcile-unrealized-casts --convert-to-wasmstack -verify-wasmstack 2>&1 | FileCheck %s --check-prefix=VERIFY

// Tests for math dialect to WasmSSA conversion
// Currently only math.sqrt is supported

// Verify no unrealized conversion casts remain
// CHECK-NOT: unrealized_conversion_cast
// VERIFY-NOT: error

//===----------------------------------------------------------------------===//
// Basic sqrt operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_sqrt_f32
func.func @test_sqrt_f32(%arg: f32) -> f32 {
  // CHECK: wasmssa.sqrt %{{.*}} : f32
  %result = math.sqrt %arg : f32
  return %result : f32
}

// CHECK-LABEL: func.func @test_sqrt_f64
func.func @test_sqrt_f64(%arg: f64) -> f64 {
  // CHECK: wasmssa.sqrt %{{.*}} : f64
  %result = math.sqrt %arg : f64
  return %result : f64
}

//===----------------------------------------------------------------------===//
// sqrt in expressions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_sqrt_in_expression
func.func @test_sqrt_in_expression(%a: f32, %b: f32) -> f32 {
  // Compute sqrt(a*a + b*b) - Pythagorean theorem
  %a2 = arith.mulf %a, %a : f32
  %b2 = arith.mulf %b, %b : f32
  %sum = arith.addf %a2, %b2 : f32
  // CHECK: wasmssa.sqrt
  %result = math.sqrt %sum : f32
  return %result : f32
}

// CHECK-LABEL: func.func @test_sqrt_chain
func.func @test_sqrt_chain(%arg: f32) -> f32 {
  // sqrt(sqrt(x))
  // CHECK: wasmssa.sqrt
  // CHECK: wasmssa.sqrt
  %s1 = math.sqrt %arg : f32
  %s2 = math.sqrt %s1 : f32
  return %s2 : f32
}

//===----------------------------------------------------------------------===//
// sqrt combined with other operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_sqrt_and_arithmetic
func.func @test_sqrt_and_arithmetic(%a: f32, %b: f32) -> f32 {
  // Compute (sqrt(a) + sqrt(b)) / 2
  %c2 = arith.constant 2.0 : f32
  // CHECK: wasmssa.sqrt
  %sa = math.sqrt %a : f32
  // CHECK: wasmssa.sqrt
  %sb = math.sqrt %b : f32
  %sum = arith.addf %sa, %sb : f32
  %result = arith.divf %sum, %c2 : f32
  return %result : f32
}

// CHECK-LABEL: func.func @test_sqrt_multi_use
func.func @test_sqrt_multi_use(%x: f32) -> f32 {
  // sqrt(x) used multiple times
  // CHECK: wasmssa.sqrt
  %s = math.sqrt %x : f32
  %r1 = arith.mulf %s, %s : f32  // s^2 = x
  %r2 = arith.addf %r1, %s : f32  // x + sqrt(x)
  return %r2 : f32
}

//===----------------------------------------------------------------------===//
// Complex patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_distance_2d
func.func @test_distance_2d(%x1: f32, %y1: f32, %x2: f32, %y2: f32) -> f32 {
  // Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)
  %dx = arith.subf %x2, %x1 : f32
  %dy = arith.subf %y2, %y1 : f32
  %dx2 = arith.mulf %dx, %dx : f32
  %dy2 = arith.mulf %dy, %dy : f32
  %sum = arith.addf %dx2, %dy2 : f32
  // CHECK: wasmssa.sqrt
  %dist = math.sqrt %sum : f32
  return %dist : f32
}

// CHECK-LABEL: func.func @test_normalize_vector
func.func @test_normalize_vector(%x: f32, %y: f32) -> (f32, f32) {
  // Normalize 2D vector
  %x2 = arith.mulf %x, %x : f32
  %y2 = arith.mulf %y, %y : f32
  %len2 = arith.addf %x2, %y2 : f32
  // CHECK: wasmssa.sqrt
  %len = math.sqrt %len2 : f32
  %nx = arith.divf %x, %len : f32
  %ny = arith.divf %y, %len : f32
  return %nx, %ny : f32, f32
}

//===----------------------------------------------------------------------===//
// sqrt edge cases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_sqrt_constant
func.func @test_sqrt_constant() -> f32 {
  %c4 = arith.constant 4.0 : f32
  // CHECK: wasmssa.sqrt
  %result = math.sqrt %c4 : f32
  return %result : f32
}

// CHECK-LABEL: func.func @test_sqrt_zero
func.func @test_sqrt_zero() -> f32 {
  %c0 = arith.constant 0.0 : f32
  // CHECK: wasmssa.sqrt
  %result = math.sqrt %c0 : f32
  return %result : f32
}

// CHECK-LABEL: func.func @test_sqrt_one
func.func @test_sqrt_one() -> f64 {
  %c1 = arith.constant 1.0 : f64
  // CHECK: wasmssa.sqrt
  %result = math.sqrt %c1 : f64
  return %result : f64
}
