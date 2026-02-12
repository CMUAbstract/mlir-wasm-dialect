// RUN: not wasm-opt %s --wami-convert-math 2>&1 | FileCheck %s

func.func @unsupported_math(%x: f32) -> f32 {
  %y = math.exp %x : f32
  return %y : f32
}

// CHECK: failed to legalize operation 'math.exp'
