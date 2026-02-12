// RUN: wasm-opt %s --wami-convert-math | FileCheck %s

// CHECK-LABEL: func @sqrt_f32
func.func @sqrt_f32(%x: f32) -> f32 {
  // CHECK: wasmssa.sqrt %{{.*}} : f32
  %y = math.sqrt %x : f32
  return %y : f32
}

// CHECK-LABEL: func @sqrt_f64
func.func @sqrt_f64(%x: f64) -> f64 {
  // CHECK: wasmssa.sqrt %{{.*}} : f64
  %y = math.sqrt %x : f64
  return %y : f64
}
