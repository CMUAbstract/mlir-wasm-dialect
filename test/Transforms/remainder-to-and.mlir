// RUN: wasm-opt %s --remainder-to-and | FileCheck %s

func.func private @use_i32(i32) -> ()

//===----------------------------------------------------------------------===//
// Positive: remui with power-of-2 -> converted to andi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remui_power_of_2
// CHECK-SAME:    (%[[X:.*]]: i32)
// CHECK:         %[[MASK:.*]] = arith.constant 3 : i32
// CHECK:         %[[AND:.*]] = arith.andi %[[X]], %[[MASK]] : i32
// CHECK:         return %[[AND]] : i32
func.func @remui_power_of_2(%x: i32) -> i32 {
  %c4 = arith.constant 4 : i32
  %rem = arith.remui %x, %c4 : i32
  return %rem : i32
}

//===----------------------------------------------------------------------===//
// Positive: remui with larger power-of-2 (8) -> mask is 7
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remui_power_of_2_larger
// CHECK-SAME:    (%[[X:.*]]: i32)
// CHECK:         %[[MASK:.*]] = arith.constant 7 : i32
// CHECK:         %[[AND:.*]] = arith.andi %[[X]], %[[MASK]] : i32
// CHECK:         return %[[AND]] : i32
func.func @remui_power_of_2_larger(%x: i32) -> i32 {
  %c8 = arith.constant 8 : i32
  %rem = arith.remui %x, %c8 : i32
  return %rem : i32
}

//===----------------------------------------------------------------------===//
// Positive: remsi with loop IV (provably non-negative) -> converted
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remsi_nonneg_loop_iv
// CHECK:         scf.for
// CHECK-NOT:       arith.remsi
// CHECK:           arith.andi
// CHECK:         }
func.func @remsi_nonneg_loop_iv() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c500 = arith.constant 500 : index
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %c500 step %c1 {
    %cast = arith.index_cast %i : index to i32
    %rem = arith.remsi %cast, %c4 : i32
    func.call @use_i32(%rem) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Positive: remsi with IV + constant offset (nussinov pattern) -> converted
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remsi_nonneg_iv_plus_offset
// CHECK:         scf.for
// CHECK-NOT:       arith.remsi
// CHECK:           arith.andi
// CHECK:         }
func.func @remsi_nonneg_iv_plus_offset() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c500 = arith.constant 500 : index
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  scf.for %i = %c0 to %c500 step %c1 {
    %cast = arith.index_cast %i : index to i32
    %plus1 = arith.addi %cast, %c1_i32 : i32
    %rem = arith.remsi %plus1, %c4 : i32
    func.call @use_i32(%rem) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Negative: remsi with unknown-sign LHS -> NOT converted
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remsi_unknown_sign
// CHECK:         arith.remsi
// CHECK:         return
func.func @remsi_unknown_sign(%x: i32) -> i32 {
  %c4 = arith.constant 4 : i32
  %rem = arith.remsi %x, %c4 : i32
  return %rem : i32
}

//===----------------------------------------------------------------------===//
// Negative: non-power-of-2 divisor -> NOT converted
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remui_non_power_of_2
// CHECK:         arith.remui
// CHECK:         return
func.func @remui_non_power_of_2(%x: i32) -> i32 {
  %c5 = arith.constant 5 : i32
  %rem = arith.remui %x, %c5 : i32
  return %rem : i32
}

// CHECK-LABEL: func.func @remsi_non_power_of_2
// CHECK:         arith.remsi
// CHECK:         return
func.func @remsi_non_power_of_2(%x: i32) -> i32 {
  %c3 = arith.constant 3 : i32
  %rem = arith.remsi %x, %c3 : i32
  return %rem : i32
}

//===----------------------------------------------------------------------===//
// Negative: divisor is 1 -> NOT converted (leave for constant fold)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remui_by_one
// CHECK:         arith.remui
// CHECK:         return
func.func @remui_by_one(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %rem = arith.remui %x, %c1 : i32
  return %rem : i32
}

// CHECK-LABEL: func.func @remsi_by_one
// CHECK:         arith.remsi
// CHECK:         return
func.func @remsi_by_one(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %rem = arith.remsi %x, %c1 : i32
  return %rem : i32
}
