// RUN: wasm-opt %s --strength-reduce | FileCheck %s

func.func private @use_i32(i32) -> ()
func.func private @use_index(index) -> ()

//===----------------------------------------------------------------------===//
// Test 1: Single addi(acc, base) user — base absorbed into init
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_single_user
// CHECK-SAME:    (%{{.*}}: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for {{.*}} iter_args(%[[ACC1:.*]] = %[[BASE]], %{{.*}} = %{{.*}}) -> (i32, i32) {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC1]])
// CHECK:           call @use_i32(%{{.*}})
// CHECK:           arith.addi %[[ACC1]]
// CHECK:           arith.addi
// CHECK:           scf.yield
// CHECK:         }
func.func @absorb_single_user(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul1 = arith.muli %cast, %c4 : i32
    %mul2 = arith.muli %cast, %c8 : i32
    %addr = arith.addi %mul1, %base : i32
    func.call @use_i32(%addr) : (i32) -> ()
    func.call @use_i32(%mul2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 2: Common base with different deltas — base absorbed, deltas remain
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_common_base_deltas
// CHECK-SAME:    (%{{.*}}: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for {{.*}} iter_args(%[[ACC1:.*]] = %[[BASE]], %{{.*}} = %{{.*}}) -> (i32, i32) {
// CHECK-NOT:       arith.muli
// CHECK:           %[[ADDR1:.*]] = arith.addi %[[ACC1]], %{{.*}} : i32
// CHECK:           call @use_i32(%[[ACC1]])
// CHECK:           call @use_i32(%[[ADDR1]])
// CHECK:           call @use_i32(%{{.*}})
// CHECK:         }
func.func @absorb_common_base_deltas(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c4_off = arith.constant 4 : i32
  %base_plus_4 = arith.addi %base, %c4_off : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul1 = arith.muli %cast, %c4 : i32
    %mul2 = arith.muli %cast, %c8 : i32
    %addr0 = arith.addi %mul1, %base : i32
    %addr1 = arith.addi %mul1, %base_plus_4 : i32
    func.call @use_i32(%addr0) : (i32) -> ()
    func.call @use_i32(%addr1) : (i32) -> ()
    func.call @use_i32(%mul2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 3: Multiple users, all with same invariant — all absorbed
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_all_same_invariant
// CHECK-SAME:    (%{{.*}}: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for {{.*}} iter_args(%[[ACC1:.*]] = %[[BASE]], %{{.*}} = %{{.*}}) -> (i32, i32) {
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC1]])
// CHECK:           call @use_i32(%[[ACC1]])
// CHECK:           call @use_i32(%{{.*}})
// CHECK:         }
func.func @absorb_all_same_invariant(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul1 = arith.muli %cast, %c4 : i32
    %mul2 = arith.muli %cast, %c8 : i32
    %addr0 = arith.addi %mul1, %base : i32
    %addr1 = arith.addi %mul1, %base : i32
    func.call @use_i32(%addr0) : (i32) -> ()
    func.call @use_i32(%addr1) : (i32) -> ()
    func.call @use_i32(%mul2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Negative — non-addi use of accumulator blocks absorption
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_negative_mixed_use
// CHECK:         scf.for {{.*}} iter_args(%[[ACC1:.*]] = %{{.*}}, %[[ACC2:.*]] = %{{.*}}) -> (i32, i32) {
// CHECK-NOT:       arith.muli
// CHECK:           %[[ADDR:.*]] = arith.addi %[[ACC1]], %{{.*}} : i32
// CHECK:           call @use_i32(%[[ADDR]])
// CHECK:           call @use_i32(%[[ACC1]])
// CHECK:           call @use_i32(%[[ACC2]])
// CHECK:         }
func.func @absorb_negative_mixed_use(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul1 = arith.muli %cast, %c4 : i32
    %mul2 = arith.muli %cast, %c8 : i32
    %addr = arith.addi %mul1, %base : i32
    func.call @use_i32(%addr) : (i32) -> ()
    func.call @use_i32(%mul1) : (i32) -> ()
    func.call @use_i32(%mul2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Negative — different bases block absorption
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_negative_different_bases
// CHECK:         scf.for {{.*}} iter_args(%[[ACC1:.*]] = %{{.*}}, %[[ACC2:.*]] = %{{.*}}) -> (i32, i32) {
// CHECK-NOT:       arith.muli
// CHECK:           %[[ADDR0:.*]] = arith.addi %[[ACC1]], %{{.*}} : i32
// CHECK:           %[[ADDR1:.*]] = arith.addi %[[ACC1]], %{{.*}} : i32
// CHECK:           call @use_i32(%[[ADDR0]])
// CHECK:           call @use_i32(%[[ADDR1]])
// CHECK:           call @use_i32(%[[ACC2]])
// CHECK:         }
func.func @absorb_negative_different_bases(%n: index, %base1: i32, %base2: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.for %i = %c0 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul1 = arith.muli %cast, %c4 : i32
    %mul2 = arith.muli %cast, %c8 : i32
    %addr0 = arith.addi %mul1, %base1 : i32
    %addr1 = arith.addi %mul1, %base2 : i32
    func.call @use_i32(%addr0) : (i32) -> ()
    func.call @use_i32(%addr1) : (i32) -> ()
    func.call @use_i32(%mul2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 6: Non-zero lower bound — init = lb*factor + base
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_nonzero_lb
// CHECK-SAME:    (%{{.*}}: index, %[[BASE:.*]]: i32)
// CHECK:         %[[INIT1:.*]] = arith.addi %{{.*}}, %[[BASE]] : i32
// CHECK:         scf.for {{.*}} iter_args(%[[ACC1:.*]] = %[[INIT1]], %{{.*}} = %{{.*}}) -> (i32, i32)
// CHECK-NOT:       arith.muli
// CHECK:           call @use_i32(%[[ACC1]])
// CHECK:         }
func.func @absorb_nonzero_lb(%n: index, %base: i32) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.for %i = %c2 to %n step %c1 {
    %cast = arith.index_cast %i : index to i32
    %mul1 = arith.muli %cast, %c4 : i32
    %mul2 = arith.muli %cast, %c8 : i32
    %addr = arith.addi %mul1, %base : i32
    func.call @use_i32(%addr) : (i32) -> ()
    func.call @use_i32(%mul2) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 7: Deep decomposition — unrolled muli(cast(K), 4) pattern
//
// After 4x unrolling, invariants differ by constants hidden inside muli:
//   addi(acc, addi(muli(cast(K), 4), base))   delta=0
//   addi(acc, addi(muli(cast(K+1), 4), base)) delta=4
//   addi(acc, addi(muli(cast(K+2), 4), base)) delta=8
//   addi(acc, addi(muli(cast(K+3), 4), base)) delta=12
// getConstantDifference walks through addi→muli→cast to find deltas.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_deep_muli_unroll
// CHECK-SAME:    (%{{.*}}: index, %[[K:.*]]: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32) {
// CHECK-NOT:       arith.muli
// CHECK:           %[[A0:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK:           %[[A1:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK:           %[[A2:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           call @use_i32(%[[A0]])
// CHECK:           call @use_i32(%[[A1]])
// CHECK:           call @use_i32(%[[A2]])
// CHECK:         }
func.func @absorb_deep_muli_unroll(%n: index, %k: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %k1 = arith.addi %k, %c1 : index
  %c2 = arith.constant 2 : index
  %k2 = arith.addi %k, %c2 : index
  %c3 = arith.constant 3 : index
  %k3 = arith.addi %k, %c3 : index
  // Precompute invariants: addi(muli(cast(K+i), 4), base)
  %ck0 = arith.index_cast %k : index to i32
  %m0 = arith.muli %ck0, %c4_i32 : i32
  %inv0 = arith.addi %m0, %base : i32
  %ck1 = arith.index_cast %k1 : index to i32
  %m1 = arith.muli %ck1, %c4_i32 : i32
  %inv1 = arith.addi %m1, %base : i32
  %ck2 = arith.index_cast %k2 : index to i32
  %m2 = arith.muli %ck2, %c4_i32 : i32
  %inv2 = arith.addi %m2, %base : i32
  %ck3 = arith.index_cast %k3 : index to i32
  %m3 = arith.muli %ck3, %c4_i32 : i32
  %inv3 = arith.addi %m3, %base : i32
  scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %a0 = arith.addi %acc, %inv0 : i32
    %a1 = arith.addi %acc, %inv1 : i32
    %a2 = arith.addi %acc, %inv2 : i32
    %a3 = arith.addi %acc, %inv3 : i32
    func.call @use_i32(%a0) : (i32) -> ()
    func.call @use_i32(%a1) : (i32) -> ()
    func.call @use_i32(%a2) : (i32) -> ()
    func.call @use_i32(%a3) : (i32) -> ()
    %next = arith.addi %acc, %c16_i32 : i32
    scf.yield %next : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 8: Deep decomposition with cast chain — muli through cast
//
// Similar to Test 7 but with unrealized_conversion_cast wrapping index_cast.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_deep_muli_cast_chain
// CHECK-SAME:    (%{{.*}}: index, %[[K:.*]]: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32) {
// CHECK-NOT:       arith.muli
// CHECK:           %[[A0:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           call @use_i32(%[[A0]])
// CHECK:         }
func.func @absorb_deep_muli_cast_chain(%n: index, %k: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c8_i32 = arith.constant 8 : i32
  %k1 = arith.addi %k, %c1 : index
  // cast chain: unrealized_conversion_cast(index_cast(k))
  %ck0_idx = arith.index_cast %k : index to i32
  %ck0 = builtin.unrealized_conversion_cast %ck0_idx : i32 to i32
  %m0 = arith.muli %ck0, %c4_i32 : i32
  %inv0 = arith.addi %m0, %base : i32
  %ck1_idx = arith.index_cast %k1 : index to i32
  %ck1 = builtin.unrealized_conversion_cast %ck1_idx : i32 to i32
  %m1 = arith.muli %ck1, %c4_i32 : i32
  %inv1 = arith.addi %m1, %base : i32
  scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %a0 = arith.addi %acc, %inv0 : i32
    %a1 = arith.addi %acc, %inv1 : i32
    func.call @use_i32(%a0) : (i32) -> ()
    func.call @use_i32(%a1) : (i32) -> ()
    %next = arith.addi %acc, %c8_i32 : i32
    scf.yield %next : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 9: Negative — different muli factors block absorption
//
// addi(acc, muli(cast(K), 4)) vs addi(acc, muli(cast(K+1), 8))
// Different factors (4 vs 8) → getConstantDifference returns nullopt.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_negative_different_factors
// CHECK:         scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32) {
// CHECK:           %[[A0:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK:           %[[A1:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK:           call @use_i32(%[[A0]])
// CHECK:           call @use_i32(%[[A1]])
// CHECK:         }
func.func @absorb_negative_different_factors(%n: index, %k: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4_i32 = arith.constant 4 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %c12_i32 = arith.constant 12 : i32
  %k1 = arith.addi %k, %c1 : index
  %ck0 = arith.index_cast %k : index to i32
  %m0 = arith.muli %ck0, %c4_i32 : i32
  %ck1 = arith.index_cast %k1 : index to i32
  %m1 = arith.muli %ck1, %c8_i32 : i32
  scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %a0 = arith.addi %acc, %m0 : i32
    %a1 = arith.addi %acc, %m1 : i32
    func.call @use_i32(%a0) : (i32) -> ()
    func.call @use_i32(%a1) : (i32) -> ()
    %next = arith.addi %acc, %c12_i32 : i32
    scf.yield %next : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 10: Deep decomposition through subi — invariants differ via subi
//
// inv0 = subi(base, offset0), inv1 = subi(base, offset1)
// getConstantDifference walks through subi with shared lhs.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_deep_subi
// CHECK-SAME:    (%{{.*}}: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32) {
// CHECK:           %[[A0:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK:           call @use_i32(%[[A0]])
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:         }
func.func @absorb_deep_subi(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c10_i32 = arith.constant 10 : i32
  %c14_i32 = arith.constant 14 : i32
  %c8_i32 = arith.constant 8 : i32
  // inv0 = base - 10, inv1 = base - 14  →  diff = 4
  %inv0 = arith.subi %base, %c10_i32 : i32
  %inv1 = arith.subi %base, %c14_i32 : i32
  scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %a0 = arith.addi %acc, %inv0 : i32
    %a1 = arith.addi %acc, %inv1 : i32
    func.call @use_i32(%a0) : (i32) -> ()
    func.call @use_i32(%a1) : (i32) -> ()
    %next = arith.addi %acc, %c8_i32 : i32
    scf.yield %next : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 11: Asymmetric addi — one invariant is base, the other is addi(base, 4)
//
// getConstantDifference(addi(base, 4), base) uses the asymmetric addi path.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @absorb_asymmetric_addi
// CHECK-SAME:    (%{{.*}}: index, %[[BASE:.*]]: i32)
// CHECK:         scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (i32) {
// CHECK:           %[[A0:.*]] = arith.addi %[[ACC]], %{{.*}} : i32
// CHECK:           call @use_i32(%[[ACC]])
// CHECK:           call @use_i32(%[[A0]])
// CHECK:         }
func.func @absorb_asymmetric_addi(%n: index, %base: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c8_i32 = arith.constant 8 : i32
  // inv0 = base, inv1 = addi(base, 4)  →  diff = 4
  %inv1 = arith.addi %base, %c4_i32 : i32
  scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    %a0 = arith.addi %acc, %base : i32
    %a1 = arith.addi %acc, %inv1 : i32
    func.call @use_i32(%a0) : (i32) -> ()
    func.call @use_i32(%a1) : (i32) -> ()
    %next = arith.addi %acc, %c8_i32 : i32
    scf.yield %next : i32
  }
  return
}
