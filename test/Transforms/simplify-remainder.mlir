// RUN: wasm-opt %s --simplify-remainder | FileCheck %s

func.func private @use_i32(i32) -> ()

//===----------------------------------------------------------------------===//
// Power-of-2 AND: remui(x, 2^k) -> andi(x, 2^k - 1)
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
// Power-of-2 AND: remsi(x, 2^k) -> andi(x, 2^k - 1) when x >= 0
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
// Eliminate: remsi(x, N) -> x when 0 <= x < N
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remsi_eliminate
// CHECK:         scf.for
// CHECK-NOT:       arith.remsi
// CHECK-NOT:       arith.remui
// CHECK-NOT:       arith.select
// CHECK:           %[[CAST:.*]] = arith.index_cast
// CHECK:           func.call @use_i32(%[[CAST]])
// CHECK:         }
func.func @remsi_eliminate() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c400 = arith.constant 400 : index
  %c400_i32 = arith.constant 400 : i32
  scf.for %i = %c0 to %c400 step %c1 {
    // IV range is [0, 399], so remsi(iv, 400) == iv
    %cast = arith.index_cast %i : index to i32
    %rem = arith.remsi %cast, %c400_i32 : i32
    func.call @use_i32(%rem) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Select: remsi(x, N) -> select(x < N, x, x - N) when 0 <= x < 2N
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remsi_select
// CHECK:         scf.for
// CHECK-NOT:       arith.remsi
// CHECK-NOT:       arith.remui
// CHECK:           %[[VAL:.*]] = arith.addi
// CHECK:           %[[CMP:.*]] = arith.cmpi slt, %[[VAL]]
// CHECK:           %[[SUB:.*]] = arith.subi %[[VAL]]
// CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[VAL]], %[[SUB]]
// CHECK:           func.call @use_i32(%[[SEL]])
// CHECK:         }
func.func @remsi_select() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c400 = arith.constant 400 : index
  %c3_i32 = arith.constant 3 : i32
  %c400_i32 = arith.constant 400 : i32
  scf.for %i = %c0 to %c400 step %c1 {
    // IV range is [0, 399], so (iv + 3) range is [3, 402].
    // 402 < 2*400 = 800, so select pattern applies.
    %cast = arith.index_cast %i : index to i32
    %plus3 = arith.addi %cast, %c3_i32 : i32
    %rem = arith.remsi %plus3, %c400_i32 : i32
    func.call @use_i32(%rem) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Signed->unsigned: remsi(x, N) -> remui(x, N) when x >= 0
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remsi_to_remui
// CHECK:         scf.for
// CHECK-NOT:       arith.remsi
// CHECK:           arith.remui
// CHECK:         }
func.func @remsi_to_remui() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1000 = arith.constant 1000 : index
  %c3_i32 = arith.constant 3 : i32
  %c400_i32 = arith.constant 400 : i32
  scf.for %i = %c0 to %c1000 step %c1 {
    // IV range is [0, 999], so (iv * 3) range is [0, 2997].
    // 2997 >= 2*400 = 800, so select doesn't apply. But smin >= 0, so
    // signed->unsigned conversion applies.
    %cast = arith.index_cast %i : index to i32
    %mul = arith.muli %cast, %c3_i32 : i32
    %rem = arith.remsi %mul, %c400_i32 : i32
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
// Negative: non-power-of-2 remui -> NOT converted
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @remui_non_power_of_2
// CHECK:         arith.remui
// CHECK:         return
func.func @remui_non_power_of_2(%x: i32) -> i32 {
  %c5 = arith.constant 5 : i32
  %rem = arith.remui %x, %c5 : i32
  return %rem : i32
}

// CHECK-LABEL: func.func @remsi_non_power_of_2_unknown
// CHECK:         arith.remsi
// CHECK:         return
func.func @remsi_non_power_of_2_unknown(%x: i32) -> i32 {
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
