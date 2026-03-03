// RUN: wasm-opt %s --hoist-from-scf-if | FileCheck %s

func.func private @use(i32) -> ()
func.func private @produce() -> i32

//===----------------------------------------------------------------------===//
// Test 1: Basic constant hoisted out of scf.if
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @basic_constant
// CHECK:         %[[C:.*]] = arith.constant 2000 : i32
// CHECK:         scf.if
// CHECK-NOT:       arith.constant
// CHECK:           func.call @use(%[[C]])
// CHECK:         }
func.func @basic_constant(%cond: i1) {
  scf.if %cond {
    %c = arith.constant 2000 : i32
    func.call @use(%c) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 2: Chain of pure ops (constant + muli + addi) all hoisted
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @chain_hoisted
// CHECK:         %[[C:.*]] = arith.constant 2000 : i32
// CHECK:         %[[MUL:.*]] = arith.muli %arg1, %[[C]] : i32
// CHECK:         %[[ADD:.*]] = arith.addi %[[MUL]], %arg2 : i32
// CHECK:         scf.if
// CHECK-NOT:       arith.constant
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.addi
// CHECK:           func.call @use(%[[ADD]])
// CHECK:         }
func.func @chain_hoisted(%cond: i1, %x: i32, %y: i32) {
  scf.if %cond {
    %c = arith.constant 2000 : i32
    %mul = arith.muli %x, %c : i32
    %add = arith.addi %mul, %y : i32
    func.call @use(%add) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 3: Both then and else regions hoisted
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @both_regions
// CHECK-DAG:     %[[C1:.*]] = arith.constant 2000 : i32
// CHECK-DAG:     %[[C2:.*]] = arith.constant 3000 : i32
// CHECK:         scf.if
// CHECK-NOT:       arith.constant
// CHECK:           func.call @use(%[[C1]])
// CHECK:         } else {
// CHECK-NOT:       arith.constant
// CHECK:           func.call @use(%[[C2]])
// CHECK:         }
func.func @both_regions(%cond: i1) {
  scf.if %cond {
    %c = arith.constant 2000 : i32
    func.call @use(%c) : (i32) -> ()
  } else {
    %c = arith.constant 3000 : i32
    func.call @use(%c) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Nested scf.if — inner ops reach outer scope
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nested_if
// CHECK:         %[[C:.*]] = arith.constant 42 : i32
// CHECK:         scf.if
// CHECK:           scf.if
// CHECK-NOT:         arith.constant
// CHECK:             func.call @use(%[[C]])
// CHECK:           }
// CHECK:         }
func.func @nested_if(%cond1: i1, %cond2: i1) {
  scf.if %cond1 {
    scf.if %cond2 {
      %c = arith.constant 42 : i32
      func.call @use(%c) : (i32) -> ()
    }
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Negative — side-effecting op stays inside scf.if
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @side_effect_stays
// CHECK:         scf.if
// CHECK:           %[[V:.*]] = func.call @produce()
// CHECK:           func.call @use(%[[V]])
// CHECK:         }
func.func @side_effect_stays(%cond: i1) {
  scf.if %cond {
    %v = func.call @produce() : () -> i32
    func.call @use(%v) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 6: Negative — op depending on value defined inside region stays
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @internal_operand_stays
// CHECK:         scf.if
// CHECK:           %[[V:.*]] = func.call @produce()
// CHECK:           %[[ADD:.*]] = arith.addi %[[V]], %arg1 : i32
// CHECK:           func.call @use(%[[ADD]])
// CHECK:         }
func.func @internal_operand_stays(%cond: i1, %x: i32) {
  scf.if %cond {
    %v = func.call @produce() : () -> i32
    %add = arith.addi %v, %x : i32
    func.call @use(%add) : (i32) -> ()
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 7: scf.if with results — hoisted op still feeds scf.yield
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @if_with_results
// CHECK:         %[[C:.*]] = arith.constant 42 : i32
// CHECK:         %[[MUL:.*]] = arith.muli %arg1, %[[C]] : i32
// CHECK:         %[[RES:.*]] = scf.if
// CHECK-NOT:       arith.constant
// CHECK-NOT:       arith.muli
// CHECK:           scf.yield %[[MUL]] : i32
// CHECK:         } else {
// CHECK:           scf.yield %arg1 : i32
// CHECK:         }
// CHECK:         return %[[RES]] : i32
func.func @if_with_results(%cond: i1, %x: i32) -> i32 {
  %res = scf.if %cond -> (i32) {
    %c = arith.constant 42 : i32
    %mul = arith.muli %x, %c : i32
    scf.yield %mul : i32
  } else {
    scf.yield %x : i32
  }
  return %res : i32
}
