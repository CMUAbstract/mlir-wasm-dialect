// RUN: wasm-opt %s --wami-convert-arith | FileCheck %s

// CHECK-LABEL: func @test_add_i32
func.func @test_add_i32(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.add %{{.*}} %{{.*}} : i32
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_add_i64
func.func @test_add_i64(%arg0: i64, %arg1: i64) -> i64 {
  // CHECK: wasmssa.add %{{.*}} %{{.*}} : i64
  %0 = arith.addi %arg0, %arg1 : i64
  return %0 : i64
}

// CHECK-LABEL: func @test_sub_i32
func.func @test_sub_i32(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.sub %{{.*}} %{{.*}} : i32
  %0 = arith.subi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_mul_i32
func.func @test_mul_i32(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.mul %{{.*}} %{{.*}} : i32
  %0 = arith.muli %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_div_si
func.func @test_div_si(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.div_si %{{.*}} %{{.*}} : i32
  %0 = arith.divsi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_div_ui
func.func @test_div_ui(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.div_ui %{{.*}} %{{.*}} : i32
  %0 = arith.divui %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_rem_si
func.func @test_rem_si(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.rem_si %{{.*}} %{{.*}} : i32
  %0 = arith.remsi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_rem_ui
func.func @test_rem_ui(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.rem_ui %{{.*}} %{{.*}} : i32
  %0 = arith.remui %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_and
func.func @test_and(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.and %{{.*}} %{{.*}} : i32
  %0 = arith.andi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_or
func.func @test_or(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.or %{{.*}} %{{.*}} : i32
  %0 = arith.ori %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_xor
func.func @test_xor(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.xor %{{.*}} %{{.*}} : i32
  %0 = arith.xori %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_shl
func.func @test_shl(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.shl %{{.*}} by %{{.*}} bits : i32
  %0 = arith.shli %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_shr_s
func.func @test_shr_s(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.shr_s %{{.*}} by %{{.*}} bits : i32
  %0 = arith.shrsi %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_shr_u
func.func @test_shr_u(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.shr_u %{{.*}} by %{{.*}} bits : i32
  %0 = arith.shrui %arg0, %arg1 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_const_i32
func.func @test_const_i32() -> i32 {
  // CHECK: wasmssa.const 42 : i32
  %0 = arith.constant 42 : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_const_i64
func.func @test_const_i64() -> i64 {
  // CHECK: wasmssa.const 12345678901234 : i64
  %0 = arith.constant 12345678901234 : i64
  return %0 : i64
}

// CHECK-LABEL: func @test_const_index_negative
func.func @test_const_index_negative() -> i32 {
  // CHECK: wasmssa.const -1 : i32
  %idx = arith.constant -1 : index
  %0 = arith.index_cast %idx : index to i32
  return %0 : i32
}

// CHECK-LABEL: func @test_cmpi_eq
func.func @test_cmpi_eq(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.eq %{{.*}} %{{.*}} : i32 -> i32
  %0 = arith.cmpi eq, %arg0, %arg1 : i32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpi_ne
func.func @test_cmpi_ne(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.ne %{{.*}} %{{.*}} : i32 -> i32
  %0 = arith.cmpi ne, %arg0, %arg1 : i32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpi_slt
func.func @test_cmpi_slt(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.lt_si %{{.*}} %{{.*}} : i32 -> i32
  %0 = arith.cmpi slt, %arg0, %arg1 : i32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpi_ult
func.func @test_cmpi_ult(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.lt_ui %{{.*}} %{{.*}} : i32 -> i32
  %0 = arith.cmpi ult, %arg0, %arg1 : i32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpi_sle
func.func @test_cmpi_sle(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.le_si %{{.*}} %{{.*}} : i32 -> i32
  %0 = arith.cmpi sle, %arg0, %arg1 : i32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpi_ule
func.func @test_cmpi_ule(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.le_ui %{{.*}} %{{.*}} : i32 -> i32
  %0 = arith.cmpi ule, %arg0, %arg1 : i32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpi_sgt
func.func @test_cmpi_sgt(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.gt_si %{{.*}} %{{.*}} : i32 -> i32
  %0 = arith.cmpi sgt, %arg0, %arg1 : i32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpi_ugt
func.func @test_cmpi_ugt(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.gt_ui %{{.*}} %{{.*}} : i32 -> i32
  %0 = arith.cmpi ugt, %arg0, %arg1 : i32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpi_sge
func.func @test_cmpi_sge(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.ge_si %{{.*}} %{{.*}} : i32 -> i32
  %0 = arith.cmpi sge, %arg0, %arg1 : i32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpi_uge
func.func @test_cmpi_uge(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: wasmssa.ge_ui %{{.*}} %{{.*}} : i32 -> i32
  %0 = arith.cmpi uge, %arg0, %arg1 : i32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// Floating point tests

// CHECK-LABEL: func @test_addf
func.func @test_addf(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: wasmssa.add %{{.*}} %{{.*}} : f32
  %0 = arith.addf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_subf
func.func @test_subf(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: wasmssa.sub %{{.*}} %{{.*}} : f32
  %0 = arith.subf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_mulf
func.func @test_mulf(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: wasmssa.mul %{{.*}} %{{.*}} : f32
  %0 = arith.mulf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_divf
func.func @test_divf(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: wasmssa.div %{{.*}} %{{.*}} : f32
  %0 = arith.divf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_minimumf
func.func @test_minimumf(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: wasmssa.min %{{.*}} %{{.*}} : f32
  %0 = arith.minimumf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_maximumf
func.func @test_maximumf(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: wasmssa.max %{{.*}} %{{.*}} : f32
  %0 = arith.maximumf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_const_f32
func.func @test_const_f32() -> f32 {
  // CHECK: wasmssa.const 3.140000e+00 : f32
  %0 = arith.constant 3.14 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_const_f64
func.func @test_const_f64() -> f64 {
  // CHECK: wasmssa.const 2.718280e+00 : f64
  %0 = arith.constant 2.71828 : f64
  return %0 : f64
}

// Floating-point comparison tests

// CHECK-LABEL: func @test_cmpf_oeq
func.func @test_cmpf_oeq(%arg0: f32, %arg1: f32) -> i32 {
  // CHECK: wasmssa.eq %{{.*}} %{{.*}} : f32 -> i32
  %0 = arith.cmpf oeq, %arg0, %arg1 : f32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpf_one
func.func @test_cmpf_one(%arg0: f32, %arg1: f32) -> i32 {
  // CHECK: wasmssa.ne %{{.*}} %{{.*}} : f32 -> i32
  %0 = arith.cmpf one, %arg0, %arg1 : f32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpf_olt
func.func @test_cmpf_olt(%arg0: f32, %arg1: f32) -> i32 {
  // CHECK: wasmssa.lt %{{.*}} %{{.*}} : f32 -> i32
  %0 = arith.cmpf olt, %arg0, %arg1 : f32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpf_ole
func.func @test_cmpf_ole(%arg0: f32, %arg1: f32) -> i32 {
  // CHECK: wasmssa.le %{{.*}} %{{.*}} : f32 -> i32
  %0 = arith.cmpf ole, %arg0, %arg1 : f32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpf_ogt
func.func @test_cmpf_ogt(%arg0: f32, %arg1: f32) -> i32 {
  // CHECK: wasmssa.gt %{{.*}} %{{.*}} : f32 -> i32
  %0 = arith.cmpf ogt, %arg0, %arg1 : f32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpf_oge
func.func @test_cmpf_oge(%arg0: f32, %arg1: f32) -> i32 {
  // CHECK: wasmssa.ge %{{.*}} %{{.*}} : f32 -> i32
  %0 = arith.cmpf oge, %arg0, %arg1 : f32
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// CHECK-LABEL: func @test_cmpf_oeq_f64
func.func @test_cmpf_oeq_f64(%arg0: f64, %arg1: f64) -> i32 {
  // CHECK: wasmssa.eq %{{.*}} %{{.*}} : f64 -> i32
  %0 = arith.cmpf oeq, %arg0, %arg1 : f64
  %1 = arith.extui %0 : i1 to i32
  return %1 : i32
}

// minnumf/maxnumf tests - these use conditional logic for NaN handling
// minnumf: if either is NaN, return the other (the number)

// CHECK-LABEL: func @test_minnumf
func.func @test_minnumf(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: wasmssa.ne %{{.*}} %{{.*}} : f32
  // CHECK: wasmssa.ne %{{.*}} %{{.*}} : f32
  // CHECK: wasmssa.min %{{.*}} %{{.*}} : f32
  // CHECK: wami.select
  // CHECK: wami.select
  %0 = arith.minnumf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_maxnumf
func.func @test_maxnumf(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: wasmssa.ne %{{.*}} %{{.*}} : f32
  // CHECK: wasmssa.ne %{{.*}} %{{.*}} : f32
  // CHECK: wasmssa.max %{{.*}} %{{.*}} : f32
  // CHECK: wami.select
  // CHECK: wami.select
  %0 = arith.maxnumf %arg0, %arg1 : f32
  return %0 : f32
}

// CHECK-LABEL: func @test_minnumf_f64
func.func @test_minnumf_f64(%arg0: f64, %arg1: f64) -> f64 {
  // CHECK: wasmssa.ne %{{.*}} %{{.*}} : f64
  // CHECK: wasmssa.ne %{{.*}} %{{.*}} : f64
  // CHECK: wasmssa.min %{{.*}} %{{.*}} : f64
  // CHECK: wami.select
  // CHECK: wami.select
  %0 = arith.minnumf %arg0, %arg1 : f64
  return %0 : f64
}

// arith.select tests

// CHECK-LABEL: func @test_select_i32
func.func @test_select_i32(%cond: i1, %a: i32, %b: i32) -> i32 {
  // CHECK: wami.select %{{.*}}, %{{.*}}, %{{.*}} : i32
  %0 = arith.select %cond, %a, %b : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_select_f32
func.func @test_select_f32(%cond: i1, %a: f32, %b: f32) -> f32 {
  // CHECK: wami.select %{{.*}}, %{{.*}}, %{{.*}} : f32
  %0 = arith.select %cond, %a, %b : f32
  return %0 : f32
}

// Extension and truncation tests

// CHECK-LABEL: func @test_extui_i32_to_i64
func.func @test_extui_i32_to_i64(%arg0: i32) -> i64 {
  // CHECK: wasmssa.extend_i32_u %{{.*}} to i64
  %0 = arith.extui %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: func @test_extsi_i32_to_i64
func.func @test_extsi_i32_to_i64(%arg0: i32) -> i64 {
  // CHECK: wasmssa.extend_i32_s %{{.*}} to i64
  %0 = arith.extsi %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: func @test_trunci_i64_to_i32
func.func @test_trunci_i64_to_i32(%arg0: i64) -> i32 {
  // CHECK: wasmssa.wrap %{{.*}} : i64 to i32
  %0 = arith.trunci %arg0 : i64 to i32
  return %0 : i32
}
