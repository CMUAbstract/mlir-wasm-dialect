// RUN: wasm-opt %s --strength-reduce | FileCheck %s

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
