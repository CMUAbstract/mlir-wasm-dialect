// RUN: wasm-opt %s --wami-convert-arith --wami-convert-func --reconcile-unrealized-casts --convert-to-wasmstack -verify-wasmstack 2>&1 | FileCheck %s

// Comprehensive tests for type conversions through the full pipeline
// Tests integer/float conversions, extensions, truncations, and casts

// Verify no unrealized conversion casts remain
// CHECK-NOT: unrealized_conversion_cast

//===----------------------------------------------------------------------===//
// Integer Extension Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_extsi_i32_to_i64
func.func @test_extsi_i32_to_i64(%arg: i32) -> i64 {
  %result = arith.extsi %arg : i32 to i64
  return %result : i64
}

// CHECK-LABEL: wasmstack.func @test_extui_i32_to_i64
func.func @test_extui_i32_to_i64(%arg: i32) -> i64 {
  %result = arith.extui %arg : i32 to i64
  return %result : i64
}

// CHECK-LABEL: wasmstack.func @test_extsi_chain
func.func @test_extsi_chain(%a: i32, %b: i32) -> i64 {
  // Extend, add as i64
  %a64 = arith.extsi %a : i32 to i64
  %b64 = arith.extsi %b : i32 to i64
  %sum = arith.addi %a64, %b64 : i64
  return %sum : i64
}

//===----------------------------------------------------------------------===//
// Integer Truncation Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_trunci_i64_to_i32
func.func @test_trunci_i64_to_i32(%arg: i64) -> i32 {
  %result = arith.trunci %arg : i64 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_trunci_chain
func.func @test_trunci_chain(%a: i64, %b: i64) -> i32 {
  %sum = arith.addi %a, %b : i64
  %result = arith.trunci %sum : i64 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_extend_then_truncate
func.func @test_extend_then_truncate(%arg: i32) -> i32 {
  %ext = arith.extsi %arg : i32 to i64
  %c1 = arith.constant 1 : i64
  %added = arith.addi %ext, %c1 : i64
  %result = arith.trunci %added : i64 to i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Boolean (i1) Conversions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_i1_to_i32
func.func @test_i1_to_i32(%a: i32, %b: i32) -> i32 {
  %cmp = arith.cmpi eq, %a, %b : i32
  %result = arith.extui %cmp : i1 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_i32_to_i1
func.func @test_i32_to_i1(%arg: i32) -> i32 {
  %cond = arith.trunci %arg : i32 to i1
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %result = arith.select %cond, %c1, %c0 : i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_bool_chain
func.func @test_bool_chain(%a: i32, %b: i32, %c: i32) -> i32 {
  %cmp1 = arith.cmpi slt, %a, %b : i32
  %cmp2 = arith.cmpi slt, %b, %c : i32
  %both = arith.andi %cmp1, %cmp2 : i1
  %result = arith.extui %both : i1 to i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Float to Integer Conversions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_fptosi_f32_to_i32
func.func @test_fptosi_f32_to_i32(%arg: f32) -> i32 {
  %result = arith.fptosi %arg : f32 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_fptoui_f32_to_i32
func.func @test_fptoui_f32_to_i32(%arg: f32) -> i32 {
  %result = arith.fptoui %arg : f32 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_fptosi_f64_to_i64
func.func @test_fptosi_f64_to_i64(%arg: f64) -> i64 {
  %result = arith.fptosi %arg : f64 to i64
  return %result : i64
}

// CHECK-LABEL: wasmstack.func @test_fptosi_f64_to_i32
func.func @test_fptosi_f64_to_i32(%arg: f64) -> i32 {
  %result = arith.fptosi %arg : f64 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_fptosi_f32_to_i64
func.func @test_fptosi_f32_to_i64(%arg: f32) -> i64 {
  %result = arith.fptosi %arg : f32 to i64
  return %result : i64
}

//===----------------------------------------------------------------------===//
// Integer to Float Conversions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_sitofp_i32_to_f32
func.func @test_sitofp_i32_to_f32(%arg: i32) -> f32 {
  %result = arith.sitofp %arg : i32 to f32
  return %result : f32
}

// CHECK-LABEL: wasmstack.func @test_uitofp_i32_to_f32
func.func @test_uitofp_i32_to_f32(%arg: i32) -> f32 {
  %result = arith.uitofp %arg : i32 to f32
  return %result : f32
}

// CHECK-LABEL: wasmstack.func @test_sitofp_i64_to_f64
func.func @test_sitofp_i64_to_f64(%arg: i64) -> f64 {
  %result = arith.sitofp %arg : i64 to f64
  return %result : f64
}

// CHECK-LABEL: wasmstack.func @test_sitofp_i32_to_f64
func.func @test_sitofp_i32_to_f64(%arg: i32) -> f64 {
  %result = arith.sitofp %arg : i32 to f64
  return %result : f64
}

//===----------------------------------------------------------------------===//
// Float Precision Conversions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_fpext_f32_to_f64
func.func @test_fpext_f32_to_f64(%arg: f32) -> f64 {
  %result = arith.extf %arg : f32 to f64
  return %result : f64
}

// CHECK-LABEL: wasmstack.func @test_fptrunc_f64_to_f32
func.func @test_fptrunc_f64_to_f32(%arg: f64) -> f32 {
  %result = arith.truncf %arg : f64 to f32
  return %result : f32
}

// CHECK-LABEL: wasmstack.func @test_float_precision_roundtrip
func.func @test_float_precision_roundtrip(%arg: f32) -> f32 {
  %ext = arith.extf %arg : f32 to f64
  %c1 = arith.constant 1.0 : f64
  %added = arith.addf %ext, %c1 : f64
  %result = arith.truncf %added : f64 to f32
  return %result : f32
}

//===----------------------------------------------------------------------===//
// Mixed Type Computations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_mixed_int_float
func.func @test_mixed_int_float(%i: i32, %f: f32) -> f32 {
  %fi = arith.sitofp %i : i32 to f32
  %result = arith.addf %fi, %f : f32
  return %result : f32
}

// CHECK-LABEL: wasmstack.func @test_int_from_float_comparison
func.func @test_int_from_float_comparison(%a: f32, %b: f32) -> i32 {
  %cmp = arith.cmpf olt, %a, %b : f32
  %result = arith.extui %cmp : i1 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_complex_conversion_chain
func.func @test_complex_conversion_chain(%i32val: i32, %f64val: f64) -> i64 {
  // i32 -> i64 -> f64 -> add with f64 -> i64
  %i64val = arith.extsi %i32val : i32 to i64
  %f64from_i64 = arith.sitofp %i64val : i64 to f64
  %sum = arith.addf %f64from_i64, %f64val : f64
  %result = arith.fptosi %sum : f64 to i64
  return %result : i64
}

//===----------------------------------------------------------------------===//
// Conversions in Control Flow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_conversion_in_if
func.func @test_conversion_in_if(%cond: i1, %i: i32, %f: f32) -> f32 {
  %result = scf.if %cond -> (f32) {
    %conv = arith.sitofp %i : i32 to f32
    scf.yield %conv : f32
  } else {
    scf.yield %f : f32
  }
  return %result : f32
}

// CHECK-LABEL: wasmstack.func @test_conversion_in_loop
func.func @test_conversion_in_loop(%n: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32

  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f32) {
    // Convert loop counter to float and add
    %i_i32 = arith.index_cast %i : index to i32
    %i_f32 = arith.sitofp %i_i32 : i32 to f32
    %new_acc = arith.addf %acc, %i_f32 : f32
    scf.yield %new_acc : f32
  }
  return %result : f32
}

//===----------------------------------------------------------------------===//
// Negation Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_negf
func.func @test_negf(%arg: f32) -> f32 {
  %result = arith.negf %arg : f32
  return %result : f32
}

// CHECK-LABEL: wasmstack.func @test_negf_f64
func.func @test_negf_f64(%arg: f64) -> f64 {
  %result = arith.negf %arg : f64
  return %result : f64
}

// CHECK-LABEL: wasmstack.func @test_negf_chain
func.func @test_negf_chain(%a: f32, %b: f32) -> f32 {
  %neg_a = arith.negf %a : f32
  %neg_b = arith.negf %b : f32
  %sum = arith.addf %neg_a, %neg_b : f32
  %result = arith.negf %sum : f32
  return %result : f32
}

//===----------------------------------------------------------------------===//
// Index Type Conversions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_index_to_i32
func.func @test_index_to_i32(%idx: index) -> i32 {
  %result = arith.index_cast %idx : index to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_i32_to_index
func.func @test_i32_to_index(%i: i32, %n: index) -> index {
  %idx = arith.index_cast %i : i32 to index
  %result = arith.addi %idx, %n : index
  return %result : index
}

// CHECK-LABEL: wasmstack.func @test_index_in_loop_bounds
func.func @test_index_in_loop_bounds(%start_i32: i32, %end_i32: i32) -> i32 {
  %start = arith.index_cast %start_i32 : i32 to index
  %end = arith.index_cast %end_i32 : i32 to index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32

  %result = scf.for %i = %start to %end step %c1 iter_args(%acc = %init) -> (i32) {
    %i_i32 = arith.index_cast %i : index to i32
    %new_acc = arith.addi %acc, %i_i32 : i32
    scf.yield %new_acc : i32
  }
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Float Comparisons Returning i1
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_cmpf_oeq
func.func @test_cmpf_oeq(%a: f32, %b: f32) -> i32 {
  %cmp = arith.cmpf oeq, %a, %b : f32
  %result = arith.extui %cmp : i1 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_cmpf_one
func.func @test_cmpf_one(%a: f32, %b: f32) -> i32 {
  %cmp = arith.cmpf one, %a, %b : f32
  %result = arith.extui %cmp : i1 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_cmpf_olt
func.func @test_cmpf_olt(%a: f32, %b: f32) -> i32 {
  %cmp = arith.cmpf olt, %a, %b : f32
  %result = arith.extui %cmp : i1 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_cmpf_ogt
func.func @test_cmpf_ogt(%a: f32, %b: f32) -> i32 {
  %cmp = arith.cmpf ogt, %a, %b : f32
  %result = arith.extui %cmp : i1 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_cmpf_ole
func.func @test_cmpf_ole(%a: f64, %b: f64) -> i32 {
  %cmp = arith.cmpf ole, %a, %b : f64
  %result = arith.extui %cmp : i1 to i32
  return %result : i32
}

// CHECK-LABEL: wasmstack.func @test_cmpf_oge
func.func @test_cmpf_oge(%a: f64, %b: f64) -> i32 {
  %cmp = arith.cmpf oge, %a, %b : f64
  %result = arith.extui %cmp : i1 to i32
  return %result : i32
}

//===----------------------------------------------------------------------===//
// Float Min/Max Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_minimumf
func.func @test_minimumf(%a: f32, %b: f32) -> f32 {
  %result = arith.minimumf %a, %b : f32
  return %result : f32
}

// CHECK-LABEL: wasmstack.func @test_maximumf
func.func @test_maximumf(%a: f32, %b: f32) -> f32 {
  %result = arith.maximumf %a, %b : f32
  return %result : f32
}

// CHECK-LABEL: wasmstack.func @test_minimumf_f64
func.func @test_minimumf_f64(%a: f64, %b: f64) -> f64 {
  %result = arith.minimumf %a, %b : f64
  return %result : f64
}

// CHECK-LABEL: wasmstack.func @test_maximumf_f64
func.func @test_maximumf_f64(%a: f64, %b: f64) -> f64 {
  %result = arith.maximumf %a, %b : f64
  return %result : f64
}

//===----------------------------------------------------------------------===//
// Multiple Conversions in Single Expression
//===----------------------------------------------------------------------===//

// CHECK-LABEL: wasmstack.func @test_conversion_expression
func.func @test_conversion_expression(%a: i32, %b: i64, %c: f32) -> f64 {
  // (a as i64 + b) as f64 + (c as f64)
  %a64 = arith.extsi %a : i32 to i64
  %sum_i64 = arith.addi %a64, %b : i64
  %sum_f64 = arith.sitofp %sum_i64 : i64 to f64
  %c64 = arith.extf %c : f32 to f64
  %result = arith.addf %sum_f64, %c64 : f64
  return %result : f64
}
