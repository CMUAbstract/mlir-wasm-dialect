// RUN: wasm-opt %s --strength-reduce="aggressive=true" | FileCheck %s

func.func private @use_i32(i32) -> ()
func.func private @use_i64(i64) -> ()
func.func private @use_index(index) -> ()

//===----------------------------------------------------------------------===//
// Test 1: Basic single multiply (lb=0, step=1)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @basic_single_multiply
// CHECK-DAG:     %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @basic_single_multiply(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 2: Non-zero lower bound
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nonzero_lower_bound
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK:         arith.index_cast %{{.*}} : index to i32
// CHECK:         %[[INIT:.*]] = arith.muli %{{.*}}, %[[C4]] : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @nonzero_lower_bound(%n: index) {
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  scf.for %i = %c5 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 3: Non-unit step
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nonunit_step
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK:         arith.index_cast %{{.*}} : index to i32
// CHECK:         %[[INC:.*]] = arith.muli %{{.*}}, %[[C4]] : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[INC]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @nonunit_step(%n: index) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %n step %c3 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Multiple multiplies in one loop
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @multiple_multiplies
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC1:.*]] = %{{.*}}, %[[ACC2:.*]] = %{{.*}}) -> (i32, i32)
// CHECK-NOT:       arith.muli
// CHECK-DAG:       call @use_i32(%[[ACC1]])
// CHECK-DAG:       call @use_i32(%[[ACC2]])
// CHECK-DAG:       arith.addi %[[ACC1]],
// CHECK-DAG:       arith.addi %[[ACC2]],
// CHECK:           scf.yield
func.func @multiple_multiplies(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul1 = arith.muli %cast, %c4 : i32
    %mul2 = arith.muli %cast, %c8 : i32
    func.call @use_i32(%mul1) : (i32) -> ()
    func.call @use_i32(%mul2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Nested loops (inner-first processing)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nested_loops
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:           arith.index_cast
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[INNER_ACC:.*]] = %{{.*}}) -> (i32)
// CHECK-NOT:         arith.muli
// CHECK:             call @use_i32(%[[INNER_ACC]])
// CHECK:             arith.addi %[[INNER_ACC]],
// CHECK:             scf.yield
func.func @nested_loops(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %cast_i = arith.index_cast %i : index to i32
    scf.for %j = %c0 to %n step %c1 {
      %cast_j = arith.index_cast %j : index to i32
      // muli(cast(inner_iv), outer_iv_cast) — outer_iv_cast is loop-invariant
      %mul = arith.muli %cast_j, %cast_i : i32
      func.call @use_i32(%mul) : (i32) -> ()
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 6: Factor on the left side (muli %factor, %cast)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @factor_on_left
// CHECK-DAG:     %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           arith.addi %[[ACC]], %[[C4]] : i32
// CHECK:           scf.yield
func.func @factor_on_left(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %c4, %cast : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 7: Negative — no multiply (no transformation)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_multiply
// CHECK:         scf.for
// CHECK-NOT:       iter_args
// CHECK:           arith.addi
// CHECK-NOT:       arith.muli
// CHECK:         }
func.func @no_multiply(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %add = arith.addi %cast, %c4 : i32
    func.call @use_i32(%add) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 8: Negative — factor is loop-variant (no transformation)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @loop_variant_factor
// CHECK:         scf.for
// CHECK-NOT:       iter_args
// CHECK:           arith.muli
// CHECK:         }
func.func @loop_variant_factor(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    // Factor is computed inside the loop (loop-variant)
    %variant = arith.addi %cast, %cast : i32
    %mul = arith.muli %cast, %variant : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 9: Existing iter_args preserved
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @existing_iter_args
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[EXISTING:.*]] = %{{.*}}, %[[ACC:.*]] = %{{.*}}) -> (i32, i32)
// CHECK-NOT:       arith.muli
// CHECK:           arith.addi %[[EXISTING]],
// CHECK:           arith.addi %[[ACC]],
// CHECK:           scf.yield
func.func @existing_iter_args(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %init = arith.constant 0 : i32
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    func.call @use_i32(%mul) : (i32) -> ()
    %next = arith.addi %acc, %c4 : i32
    scf.yield %next : i32
  }
  func.call @use_i32(%result) : (i32) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 10: Direct IV multiply (no cast, index type)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @direct_iv_multiply
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (index)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_index(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : index
// CHECK:           scf.yield %[[NEXT]] : index
func.func @direct_iv_multiply(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %n step %c1 {
    %mul = arith.muli %i, %c4 : index
    func.call @use_index(%mul) : (index) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 11: Direct IV multiply with non-zero lower bound
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @direct_iv_nonzero_lb
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[INIT:.*]] = arith.muli %{{.*}}, %[[C4]] : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (index)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_index(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : index
// CHECK:           scf.yield %[[NEXT]] : index
func.func @direct_iv_nonzero_lb(%n: index) {
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c5 to %n step %c1 {
    %mul = arith.muli %i, %c4 : index
    func.call @use_index(%mul) : (index) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 12: Basic shift left (lb=0, step=1, shift=2 → factor=4)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @basic_shift_left
// CHECK-DAG:     %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32)
// CHECK-NOT:       arith.shli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @basic_shift_left(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %shl = arith.shli %cast, %c2 : i32
    func.call @use_i32(%shl) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 13: Shift left with non-zero lb and non-unit step
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @shift_left_nonzero_lb_step
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32)
// CHECK-NOT:       arith.shli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           arith.addi %[[ACC]]
// CHECK:           scf.yield
func.func @shift_left_nonzero_lb_step(%n: index) {
  %c2_idx = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : i32
  scf.for %i = %c2_idx to %n step %c3 {
    %cast = arith.index_cast %i : index to i32
    %shl = arith.shli %cast, %c2 : i32
    func.call @use_i32(%shl) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 14: Negative — shift with non-constant amount (no transformation)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @loop_variant_shift
// CHECK:         scf.for
// CHECK-NOT:       iter_args
// CHECK:           arith.shli
// CHECK:         }
func.func @loop_variant_shift(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    // Shift amount depends on iv (non-constant)
    %shl = arith.shli %cast, %cast : i32
    func.call @use_i32(%shl) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 15: ExtSI chain (index_cast → extsi → muli)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @extsi_chain
// CHECK-DAG:     %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i64
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[C0_I64]]) -> (i64)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i64(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i64
// CHECK:           scf.yield %[[NEXT]] : i64
func.func @extsi_chain(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i64
  scf.for %i = %c0 to %n step %c1 {
    %i32 = arith.index_cast %i : index to i32
    %i64 = arith.extsi %i32 : i32 to i64
    %mul = arith.muli %i64, %c4 : i64
    func.call @use_i64(%mul) : (i64) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 16: ExtUI chain (index_cast → extui → muli)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @extui_chain
// CHECK-DAG:     %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : i64
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[C0_I64]]) -> (i64)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i64(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C8]] : i64
// CHECK:           scf.yield %[[NEXT]] : i64
func.func @extui_chain(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i64
  scf.for %i = %c0 to %n step %c1 {
    %i32 = arith.index_cast %i : index to i32
    %i64 = arith.extui %i32 : i32 to i64
    %mul = arith.muli %i64, %c8 : i64
    func.call @use_i64(%mul) : (i64) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 17: ExtSI chain with non-zero lower bound (tests cast chain recreation)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @extsi_nonzero_lb
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i64
// CHECK:         arith.index_cast
// CHECK:         arith.extsi
// CHECK:         %[[INIT:.*]] = arith.muli %{{.*}}, %[[C4]] : i64
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i64)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i64(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i64
// CHECK:           scf.yield %[[NEXT]] : i64
func.func @extsi_nonzero_lb(%n: index) {
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i64
  scf.for %i = %c5 to %n step %c1 {
    %i32 = arith.index_cast %i : index to i32
    %i64 = arith.extsi %i32 : i32 to i64
    %mul = arith.muli %i64, %c4 : i64
    func.call @use_i64(%mul) : (i64) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 18: addi(muli(cast(iv), c4), base) with lb=0 — init = base
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_basic
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 100 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[BASE]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @addi_basic(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %base = arith.constant 100 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    %addr = arith.addi %mul, %base : i32
    func.call @use_i32(%addr) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 19: addi(muli(cast(iv), c4), base) with lb=5 — init = 5*4 + base
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_nonzero_lb
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 100 : i32
// CHECK:         %[[LB_CAST:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK:         %[[MUL:.*]] = arith.muli %[[LB_CAST]], %[[C4]] : i32
// CHECK:         %[[INIT:.*]] = arith.addi %[[MUL]], %[[BASE]] : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @addi_nonzero_lb(%n: index) {
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %base = arith.constant 100 : i32
  scf.for %i = %c5 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    %addr = arith.addi %mul, %base : i32
    func.call @use_i32(%addr) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 20: addi with non-unit step — inc = step * factor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_nonunit_step
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 100 : i32
// CHECK:         %[[STEP_CAST:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK:         %[[INC:.*]] = arith.muli %[[STEP_CAST]], %[[C4]] : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[BASE]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[INC]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @addi_nonunit_step(%n: index) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : i32
  %base = arith.constant 100 : i32
  scf.for %i = %c0 to %n step %c3 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    %addr = arith.addi %mul, %base : i32
    func.call @use_i32(%addr) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 21: Offset on the left side — addi(base, muli(...))
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_offset_on_left
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 100 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[BASE]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @addi_offset_on_left(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %base = arith.constant 100 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    %addr = arith.addi %base, %mul : i32
    func.call @use_i32(%addr) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 22: shli + addi — addi(shli(cast(iv), 2), base)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_shli
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 100 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[BASE]]) -> (i32)
// CHECK-NOT:       arith.shli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @addi_shli(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : i32
  %base = arith.constant 100 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %shl = arith.shli %cast, %c2 : i32
    %addr = arith.addi %shl, %base : i32
    func.call @use_i32(%addr) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 23: Multiple addi offsets from same muli — muli fully absorbed
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_multiple_offsets
// CHECK-DAG:     %[[BASE1:.*]] = arith.constant 100 : i32
// CHECK-DAG:     %[[BASE2:.*]] = arith.constant 200 : i32
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC1:.*]] = %[[BASE1]], %[[ACC2:.*]] = %[[BASE2]]) -> (i32, i32)
// CHECK-NOT:       arith.muli
// CHECK-DAG:       call @use_i32(%[[ACC1]])
// CHECK-DAG:       call @use_i32(%[[ACC2]])
// CHECK-DAG:       arith.addi %[[ACC1]], %[[C4]]
// CHECK-DAG:       arith.addi %[[ACC2]], %[[C4]]
// CHECK:           scf.yield
func.func @addi_multiple_offsets(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %base1 = arith.constant 100 : i32
  %base2 = arith.constant 200 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    %addr1 = arith.addi %mul, %base1 : i32
    %addr2 = arith.addi %mul, %base2 : i32
    func.call @use_i32(%addr1) : (i32) -> ()
    func.call @use_i32(%addr2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 24: Mixed uses — muli has both direct use and addi use (NOT absorbed)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_mixed_uses
// CHECK-DAG:     %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 100 : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[MUL_ACC:.*]] = %[[C0_I32]], %[[ADD_ACC:.*]] = %[[BASE]]) -> (i32, i32)
// CHECK-NOT:       arith.muli
// CHECK-DAG:       call @use_i32(%[[MUL_ACC]])
// CHECK-DAG:       call @use_i32(%[[ADD_ACC]])
// CHECK-DAG:       arith.addi %[[MUL_ACC]], %[[C4]]
// CHECK-DAG:       arith.addi %[[ADD_ACC]], %[[C4]]
// CHECK:           scf.yield
func.func @addi_mixed_uses(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %base = arith.constant 100 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    func.call @use_i32(%mul) : (i32) -> ()
    %addr = arith.addi %mul, %base : i32
    func.call @use_i32(%addr) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 25: Direct IV (index type) with addi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_direct_iv
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 100 : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[BASE]]) -> (index)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_index(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C4]] : index
// CHECK:           scf.yield %[[NEXT]] : index
func.func @addi_direct_iv(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %base = arith.constant 100 : index
  scf.for %i = %c0 to %n step %c1 {
    %mul = arith.muli %i, %c4 : index
    %addr = arith.addi %mul, %base : index
    func.call @use_index(%addr) : (index) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 26: Negative — loop-variant offset (addi not transformed, muli is)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_loop_variant_offset
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           arith.addi %[[ACC]], %{{.*}} : i32
// CHECK:           arith.addi %[[ACC]], %{{.*}} : i32
// CHECK:           scf.yield
func.func @addi_loop_variant_offset(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    // Offset is loop-variant (depends on iv)
    %variant = arith.addi %cast, %cast : i32
    %addr = arith.addi %mul, %variant : i32
    func.call @use_i32(%addr) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 27: Negative — addi of two IV-dependent muli results (no addi folding)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_both_iv_dependent
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC1:.*]] = %{{.*}}, %[[ACC2:.*]] = %{{.*}}) -> (i32, i32)
// CHECK-NOT:       arith.muli
// CHECK:           %[[SUM:.*]] = arith.addi %[[ACC1]], %[[ACC2]] : i32
// CHECK:           call @use_i32(%[[SUM]])
// CHECK-DAG:       arith.addi %[[ACC1]],
// CHECK-DAG:       arith.addi %[[ACC2]],
// CHECK:           scf.yield
func.func @addi_both_iv_dependent(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul1 = arith.muli %cast, %c4 : i32
    %mul2 = arith.muli %cast, %c8 : i32
    %sum = arith.addi %mul1, %mul2 : i32
    func.call @use_i32(%sum) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 28: Phase 0 — muli(index_cast(addi(iv, k)), factor)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_cast_addi_mul
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : i32
// CHECK:         %[[INIT:.*]] = arith.muli %{{.*}}, %[[C8]] : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C8]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @distribute_cast_addi_mul(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %ip1 = arith.addi %i, %c1 : index
    %cast = arith.index_cast %ip1 : index to i32
    %mul = arith.muli %cast, %c8 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 29: Phase 0 — shli(index_cast(addi(iv, k)), n)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_cast_addi_shli
// CHECK:         arith.shli %{{.*}} : i32
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32)
// CHECK-NOT:       arith.shli
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           arith.addi %[[ACC]]
// CHECK:           scf.yield
func.func @distribute_cast_addi_shli(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : i32
  scf.for %i = %c0 to %n step %c1 {
    %ip1 = arith.addi %i, %c1 : index
    %cast = arith.index_cast %ip1 : index to i32
    %shl = arith.shli %cast, %c3 : i32
    func.call @use_i32(%shl) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 30: Phase 0 — muli(index_cast(subi(iv, k)), factor)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_subi
// CHECK:         arith.subi %{{.*}} : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           arith.addi %[[ACC]]
// CHECK:           scf.yield
func.func @distribute_subi(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %im1 = arith.subi %i, %c1 : index
    %cast = arith.index_cast %im1 : index to i32
    %mul = arith.muli %cast, %c8 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 31: Phase 0 — extsi chain: muli(extsi(index_cast(addi(iv, k))), factor)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_extsi_chain
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : i64
// CHECK:         arith.index_cast
// CHECK:         arith.extsi
// CHECK:         %[[INIT:.*]] = arith.muli %{{.*}}, %[[C8]] : i64
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i64)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i64(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C8]] : i64
// CHECK:           scf.yield %[[NEXT]] : i64
func.func @distribute_extsi_chain(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i64
  scf.for %i = %c0 to %n step %c1 {
    %ip1 = arith.addi %i, %c1 : index
    %i32 = arith.index_cast %ip1 : index to i32
    %i64 = arith.extsi %i32 : i32 to i64
    %mul = arith.muli %i64, %c8 : i64
    func.call @use_i64(%mul) : (i64) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 32: Phase 0 — jacobi-1d pattern: A[i-1], A[i], A[i+1]
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_jacobi
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A1:.*]] = %{{.*}}, %[[A2:.*]] = %{{.*}}, %[[A3:.*]] = %{{.*}}) -> (i32, i32, i32)
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.shli
// CHECK-NOT:       arith.addi
// CHECK:           call @use_i32(%[[A1]])
// CHECK:           call @use_i32(%[[A2]])
// CHECK:           call @use_i32(%[[A3]])
// CHECK:           scf.yield
func.func @distribute_jacobi(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    // (i-1) * 8 + base
    %im1 = arith.subi %i, %c1 : index
    %cast_im1 = arith.index_cast %im1 : index to i32
    %off_im1 = arith.muli %cast_im1, %c8 : i32
    %addr_im1 = arith.addi %off_im1, %base : i32
    func.call @use_i32(%addr_im1) : (i32) -> ()
    // i * 8 + base
    %cast_i = arith.index_cast %i : index to i32
    %off_i = arith.muli %cast_i, %c8 : i32
    %addr_i = arith.addi %off_i, %base : i32
    func.call @use_i32(%addr_i) : (i32) -> ()
    // (i+1) * 8 + base
    %ip1 = arith.addi %i, %c1 : index
    %cast_ip1 = arith.index_cast %ip1 : index to i32
    %off_ip1 = arith.muli %cast_ip1, %c8 : i32
    %addr_ip1 = arith.addi %off_ip1, %base : i32
    func.call @use_i32(%addr_ip1) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 33: Phase 0 — (iv+k)*factor + base (outer addi folded into init)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_with_outer_addi
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : i32
// CHECK:         %[[INIT:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:         scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C8]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
func.func @distribute_with_outer_addi(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %ip1 = arith.addi %i, %c1 : index
    %cast = arith.index_cast %ip1 : index to i32
    %mul = arith.muli %cast, %c8 : i32
    %addr = arith.addi %mul, %base : i32
    func.call @use_i32(%addr) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 34: Negative — (iv + loop_variant) * factor (no distribution)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_negative_variant
// CHECK:         scf.for
// CHECK-NOT:       iter_args
// CHECK:           arith.muli
// CHECK:         }
func.func @distribute_negative_variant(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    // Addend is loop-variant (depends on iv)
    %variant = arith.addi %cast, %cast : i32
    %mul = arith.muli %variant, %c8 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 35: Phase 0 — nested addi(subi(iv, inv), k) (nussinov pattern)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_nested_addi_subi
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           arith.addi %[[ACC]]
// CHECK:           scf.yield
func.func @distribute_nested_addi_subi(%n: index, %outer_inv: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %sub = arith.subi %i, %outer_inv : index
    %add = arith.addi %sub, %c1 : index
    %cast = arith.index_cast %add : index to i32
    %mul = arith.muli %cast, %c8 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 36: Phase 0 — nested subi(addi(iv, k), inv)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_nested_subi_addi
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           arith.addi %[[ACC]]
// CHECK:           scf.yield
func.func @distribute_nested_subi_addi(%n: index, %outer_inv: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %add = arith.addi %i, %c1 : index
    %sub = arith.subi %add, %outer_inv : index
    %cast = arith.index_cast %sub : index to i32
    %mul = arith.muli %cast, %c8 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 37: Negative — subi(inv, iv) gives negative IV coefficient (no transform)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_negative_subi_iv_rhs
// CHECK:         scf.for
// CHECK-NOT:       iter_args
// CHECK:           arith.muli
// CHECK:         }
func.func @distribute_negative_subi_iv_rhs(%n: index, %outer_inv: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %sub = arith.subi %outer_inv, %i : index
    %add = arith.addi %sub, %c1 : index
    %cast = arith.index_cast %add : index to i32
    %mul = arith.muli %cast, %c8 : i32
    func.call @use_i32(%mul) : (i32) -> ()
  }
  return
}
