// RUN: wasm-opt %s --strength-reduce | FileCheck %s

func.func private @use_i32(i32) -> ()
func.func private @use_i64(i64) -> ()
func.func private @use_index(index) -> ()

//===----------------------------------------------------------------------===//
// Test 1: Basic single multiply (lb=0, step=1)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @basic_single_multiply
// CHECK:         %[[K_IDX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:         %[[NEW_UB:.*]] = arith.muli %{{.*}}, %[[K_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %[[NEW_UB]] step %[[K_IDX]] {
// CHECK-NOT:       arith.muli
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
// CHECK:         }
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
// CHECK:         %[[K_IDX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:         %[[NEW_LB:.*]] = arith.muli %{{.*}}, %[[K_IDX]] : index
// CHECK:         %[[NEW_UB:.*]] = arith.muli %{{.*}}, %[[K_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %[[NEW_LB]] to %[[NEW_UB]] step %[[K_IDX]] {
// CHECK-NOT:       arith.muli
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
// CHECK:         }
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
// CHECK:         %[[K_IDX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:         %[[NEW_UB:.*]] = arith.muli %{{.*}}, %[[K_IDX]] : index
// CHECK:         %[[NEW_STEP:.*]] = arith.muli %{{.*}}, %[[K_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %[[NEW_UB]] step %[[NEW_STEP]] {
// CHECK-NOT:       arith.muli
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
// CHECK:         }
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
// Test 4: Nested loops (inner-first processing)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nested_loops
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           %[[CAST_I:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK:           %[[K_IDX:.*]] = arith.index_cast %[[CAST_I]] : i32 to index
// CHECK:           %[[INNER_UB:.*]] = arith.muli %{{.*}}, %[[K_IDX]] : index
// CHECK:           scf.for %[[J:.*]] = %{{.*}} to %[[INNER_UB]] step %[[K_IDX]] {
// CHECK-NOT:         arith.muli
// CHECK:             %[[CAST_J:.*]] = arith.index_cast %[[J]] : index to i32
// CHECK:             call @use_i32(%[[CAST_J]])
// CHECK:           }
// CHECK:         }
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
// Test 5: Factor on the left side (muli %factor, %cast)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @factor_on_left
// CHECK:         %[[K_IDX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:         %[[NEW_UB:.*]] = arith.muli %{{.*}}, %[[K_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %[[NEW_UB]] step %[[K_IDX]] {
// CHECK-NOT:       arith.muli
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
// CHECK:         }
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
// Test 6: Existing iter_args preserved (no new ones added)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @existing_iter_args
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[EXISTING:.*]] = %{{.*}}) -> (i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32
// CHECK:           arith.addi %[[EXISTING]],
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
// Test 7: Direct IV multiply (no cast, index type)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @direct_iv_multiply
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[NEW_UB:.*]] = arith.muli %{{.*}}, %[[C4]] : index
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %[[NEW_UB]] step %[[C4]] {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_index(%[[I]])
// CHECK:         }
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
// Test 8: Direct IV multiply with non-zero lower bound
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @direct_iv_nonzero_lb
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[NEW_LB:.*]] = arith.muli %{{.*}}, %[[C4]] : index
// CHECK:         %[[NEW_UB:.*]] = arith.muli %{{.*}}, %[[C4]] : index
// CHECK:         scf.for %[[I:.*]] = %[[NEW_LB]] to %[[NEW_UB]] step %[[C4]] {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_index(%[[I]])
// CHECK:         }
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
// Test 9: Basic shift left (lb=0, step=1, shift=2 → factor=4)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @basic_shift_left
// CHECK-DAG:     %[[C4_IDX:.*]] = arith.constant 4 : index
// CHECK:         %[[NEW_UB:.*]] = arith.muli %{{.*}}, %[[C4_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %[[NEW_UB]] step %[[C4_IDX]] {
// CHECK-NOT:       arith.shli
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
// CHECK:         }
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
// Test 10: Shift left with non-zero lb and non-unit step
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @shift_left_nonzero_lb_step
// CHECK-DAG:     %[[C4_IDX:.*]] = arith.constant 4 : index
// CHECK:         %[[NEW_LB:.*]] = arith.muli %{{.*}}, %[[C4_IDX]] : index
// CHECK:         %[[NEW_UB:.*]] = arith.muli %{{.*}}, %[[C4_IDX]] : index
// CHECK:         %[[NEW_STEP:.*]] = arith.muli %{{.*}}, %[[C4_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %[[NEW_LB]] to %[[NEW_UB]] step %[[NEW_STEP]] {
// CHECK-NOT:       arith.shli
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
// CHECK:         }
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
// Test 11: ExtSI chain (index_cast → extsi → muli)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @extsi_chain
// CHECK:         %[[K_IDX:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK:         %[[NEW_UB:.*]] = arith.muli %{{.*}}, %[[K_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %[[NEW_UB]] step %[[K_IDX]] {
// CHECK-NOT:       arith.muli
// CHECK:           %[[I32:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           %[[I64:.*]] = arith.extsi %[[I32]] : i32 to i64
// CHECK:           call @use_i64(%[[I64]])
// CHECK:         }
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
// Test 12: ExtUI chain (index_cast → extui → muli)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @extui_chain
// CHECK:         %[[K_IDX:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK:         %[[NEW_UB:.*]] = arith.muli %{{.*}}, %[[K_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %[[NEW_UB]] step %[[K_IDX]] {
// CHECK-NOT:       arith.muli
// CHECK:           %[[I32:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           %[[I64:.*]] = arith.extui %[[I32]] : i32 to i64
// CHECK:           call @use_i64(%[[I64]])
// CHECK:         }
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
// Test 13: addi(muli(cast(iv), c4), base) with lb=0 — addi absorbed into bounds
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_basic
// CHECK:         arith.muli %{{.*}} : index
// CHECK:         %[[OFF_IDX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:         arith.addi %{{.*}}, %[[OFF_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %[[OFF_IDX]] to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.addi
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
// CHECK:         }
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
// Test 14: addi(muli(cast(iv), c4), base) with lb=5
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_nonzero_lb
// CHECK:         arith.muli %{{.*}} : index
// CHECK:         arith.muli %{{.*}} : index
// CHECK:         %[[OFF_IDX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:         %[[NEW_LB:.*]] = arith.addi %{{.*}}, %[[OFF_IDX]] : index
// CHECK:         %[[NEW_UB:.*]] = arith.addi %{{.*}}, %[[OFF_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %[[NEW_LB]] to %[[NEW_UB]] step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.addi
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
// CHECK:         }
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
// Test 15: Mixed uses — muli has both direct use and addi use
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_mixed_uses
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
// CHECK:           %[[ADDR:.*]] = arith.addi %[[CAST]], %{{.*}} : i32
// CHECK:           call @use_i32(%[[ADDR]])
// CHECK:         }
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
// Test 16: Multiple addi offsets from same muli
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_multiple_offsets
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           arith.addi %[[CAST]], %{{.*}} : i32
// CHECK:           arith.addi %[[CAST]], %{{.*}} : i32
// CHECK:           call @use_i32
// CHECK:           call @use_i32
// CHECK:         }
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
// Test 17: shli + addi — addi(shli(cast(iv), 2), base)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_shli
// CHECK:         arith.muli %{{.*}} : index
// CHECK:         %[[OFF_IDX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:         arith.addi %{{.*}}, %[[OFF_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %[[OFF_IDX]] to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.shli
// CHECK-NOT:       arith.addi
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
// CHECK:         }
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
// Test 18: Direct IV (index type) with addi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @addi_direct_iv
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[BASE:.*]] = arith.constant 100 : index
// CHECK:         %[[NEW_UB:.*]] = arith.addi %{{.*}}, %[[BASE]] : index
// CHECK:         scf.for %[[I:.*]] = %[[BASE]] to %[[NEW_UB]] step %[[C4]] {
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.addi
// CHECK:           call @use_index(%[[I]])
// CHECK:         }
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
// Test 19: Phase 0 — muli(index_cast(addi(iv, k)), factor)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_cast_addi_mul
// CHECK:         arith.muli %{{.*}} : i32
// CHECK:         arith.muli %{{.*}} : index
// CHECK:         arith.index_cast %{{.*}} : i32 to index
// CHECK:         arith.addi %{{.*}} : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.addi
// CHECK:           arith.index_cast
// CHECK:           call @use_i32
// CHECK:         }
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
// Test 20: Phase 0 — shli(index_cast(addi(iv, k)), n)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_cast_addi_shli
// CHECK:         arith.shli %{{.*}} : i32
// CHECK:         arith.muli %{{.*}} : index
// CHECK:         arith.index_cast %{{.*}} : i32 to index
// CHECK:         arith.addi %{{.*}} : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.shli
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.addi
// CHECK:           arith.index_cast
// CHECK:           call @use_i32
// CHECK:         }
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
// Test 21: Phase 0 — muli(index_cast(subi(iv, k)), factor)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_subi
// CHECK:         arith.muli %{{.*}} : i32
// CHECK:         arith.muli %{{.*}} : index
// CHECK:         arith.index_cast %{{.*}} : i32 to index
// CHECK:         arith.addi %{{.*}} : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.addi
// CHECK:           arith.index_cast
// CHECK:           call @use_i32
// CHECK:         }
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
// Test 22: Phase 0 — extsi chain: muli(extsi(index_cast(addi(iv, k))), factor)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_extsi_chain
// CHECK:         arith.muli %{{.*}} : i64
// CHECK:         arith.muli %{{.*}} : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK:           arith.index_cast
// CHECK:           arith.extsi
// CHECK:           arith.addi
// CHECK:           call @use_i64
// CHECK:         }
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
// Test 23: Phase 0 — jacobi-1d pattern: A[i-1], A[i], A[i+1]
// All three multiplies share factor %c8, so bounds modification eliminates all.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_jacobi
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.shli
// CHECK:           arith.addi
// CHECK:           call @use_i32
// CHECK:           arith.addi
// CHECK:           call @use_i32
// CHECK:           arith.addi
// CHECK:           call @use_i32
// CHECK:         }
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
// Test 24: Phase 0 — (iv+k)*factor + base (offset absorbed into bounds)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_with_outer_addi
// CHECK:         arith.muli %{{.*}} : i32
// CHECK:         arith.addi %{{.*}} : i32
// CHECK:         arith.muli %{{.*}} : index
// CHECK:         arith.index_cast %{{.*}} : i32 to index
// CHECK:         arith.addi %{{.*}} : index
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.addi
// CHECK:           arith.index_cast
// CHECK:           call @use_i32
// CHECK:         }
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
// Test 25: Phase 0 — nested addi(subi(iv, inv), k)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_nested_addi_subi
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32
// CHECK:         }
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
// Test 26: Phase 0 — nested subi(addi(iv, k), inv)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_nested_subi_addi
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32
// CHECK:         }
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
// Negative tests
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test 27: Different factors — accumulator-based reduction
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @different_factors
// CHECK:         scf.for {{.*}} iter_args(%[[ACC1:.*]] = %{{.*}}, %[[ACC2:.*]] = %{{.*}}) -> (i32, i32) {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC1]])
// CHECK:           call @use_i32(%[[ACC2]])
// CHECK:           %[[NEXT1:.*]] = arith.addi %[[ACC1]]
// CHECK:           %[[NEXT2:.*]] = arith.addi %[[ACC2]]
// CHECK:           scf.yield %[[NEXT1]], %[[NEXT2]]
// CHECK:         }
func.func @different_factors(%n: index) {
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
// Test 28: IV cast still live — accumulator-based reduction
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @iv_cast_still_live
// CHECK:         scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32) {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           call @use_i32(
// CHECK:           arith.addi %[[ACC]],
// CHECK:           scf.yield
// CHECK:         }
func.func @iv_cast_still_live(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    func.call @use_i32(%mul) : (i32) -> ()
    // Extra use of cast — IV remains live after muli erasure
    func.call @use_i32(%cast) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 29: IV has direct non-candidate use — accumulator-based reduction
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @iv_direct_use
// CHECK:         scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (index) {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_index(%[[ACC]])
// CHECK:           call @use_index(
// CHECK:           arith.addi %[[ACC]],
// CHECK:           scf.yield
// CHECK:         }
func.func @iv_direct_use(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %n step %c1 {
    %mul = arith.muli %i, %c4 : index
    func.call @use_index(%mul) : (index) -> ()
    // Direct use of IV — remains live
    func.call @use_index(%i) : (index) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 30: No multiply — addi(cast(iv), offset) absorbed into bounds
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_multiply
// CHECK:         %[[OFF_IDX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:         %[[NEW_UB:.*]] = arith.addi %{{.*}}, %[[OFF_IDX]] : index
// CHECK:         scf.for %[[I:.*]] = %[[OFF_IDX]] to %[[NEW_UB]] step %{{.*}} {
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.addi
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           call @use_i32(%[[CAST]])
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
// Test 31: Negative — factor is loop-variant (no transformation)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @loop_variant_factor
// CHECK:         scf.for
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
// Test 32: Negative — shift with non-constant amount (no transformation)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @loop_variant_shift
// CHECK:         scf.for
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
// Test 33: Negative — subi(inv, iv) gives negative IV coefficient (no transform)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_negative_subi_iv_rhs
// CHECK:         scf.for
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

//===----------------------------------------------------------------------===//
// Test 34: Negative — (iv + loop_variant) * factor (no distribution)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @distribute_negative_variant
// CHECK:         scf.for
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
// Test 35: Mixed muli+shli — separate accumulators
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mixed_muli_shli
// CHECK:         scf.for {{.*}} iter_args(%[[ACC1:.*]] = %{{.*}}, %[[ACC2:.*]] = %{{.*}}) -> (i32, i32) {
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.shli
// CHECK:           call @use_i32(%[[ACC1]])
// CHECK:           call @use_i32(%[[ACC2]])
// CHECK:           arith.addi %[[ACC1]],
// CHECK:           arith.addi %[[ACC2]],
// CHECK:           scf.yield
// CHECK:         }
func.func @mixed_muli_shli(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c2 = arith.constant 2 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    %shl = arith.shli %cast, %c2 : i32
    func.call @use_i32(%mul) : (i32) -> ()
    func.call @use_i32(%shl) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 36: Accumulator with existing iter_args
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @accum_with_existing_iter_args
// CHECK:         scf.for {{.*}} iter_args(%[[SUM:.*]] = %{{.*}}, %[[ACC:.*]] = %{{.*}}) -> (i32, i32) {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           call @use_i32(
// CHECK:           arith.addi %[[SUM]]
// CHECK:           arith.addi %[[ACC]]
// CHECK:           scf.yield
// CHECK:         }
func.func @accum_with_existing_iter_args(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %init = arith.constant 0 : i32
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%sum = %init) -> (i32) {
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c4 : i32
    func.call @use_i32(%mul) : (i32) -> ()
    // Extra use of cast — prevents bounds-based
    func.call @use_i32(%cast) : (i32) -> ()
    %next = arith.addi %sum, %c8 : i32
    scf.yield %next : i32
  }
  func.call @use_i32(%result) : (i32) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 37: Accumulator with nonzero lb and non-unit step
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @accum_nonzero_lb_step
// CHECK:         scf.for {{.*}} iter_args(%[[ACC1:.*]] = %{{.*}}, %[[ACC2:.*]] = %{{.*}}) -> (i32, i32) {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC1]])
// CHECK:           call @use_i32(%[[ACC2]])
// CHECK:           arith.addi %[[ACC1]]
// CHECK:           arith.addi %[[ACC2]]
// CHECK:           scf.yield
// CHECK:         }
func.func @accum_nonzero_lb_step(%n: index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.for %i = %c2 to %n step %c3 {
    %cast = arith.index_cast %i : index to i32
    %mul1 = arith.muli %cast, %c4 : i32
    %mul2 = arith.muli %cast, %c8 : i32
    func.call @use_i32(%mul1) : (i32) -> ()
    func.call @use_i32(%mul2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 38: Standalone offset absorption (no multiply, just addi(iv, base))
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @standalone_offset_absorption
// CHECK-SAME:    (%[[N:.*]]: index, %[[BASE:.*]]: index)
// CHECK:         %[[NEW_UB:.*]] = arith.addi %[[N]], %[[BASE]] : index
// CHECK:         scf.for %[[I:.*]] = %[[BASE]] to %[[NEW_UB]] step %{{.*}} {
// CHECK-NOT:       arith.addi
// CHECK:           call @use_index(%[[I]])
// CHECK:         }
func.func @standalone_offset_absorption(%n: index, %base: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %addr = arith.addi %i, %base : index
    func.call @use_index(%addr) : (index) -> ()
  }
  return
}
