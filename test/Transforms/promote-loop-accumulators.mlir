// RUN: wasm-opt %s --promote-loop-accumulators | FileCheck %s

//===----------------------------------------------------------------------===//
// Test 1: Basic addf accumulation (the 2mm pattern)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @basic_addf_accumulation
// CHECK:         %[[INIT:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:         %[[RESULT:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f64)
// CHECK-NOT:       memref.load
// CHECK-NOT:       memref.store
// CHECK:           %[[NEW:.*]] = arith.addf %[[ACC]], %{{.*}}
// CHECK:           scf.yield %[[NEW]]
// CHECK:         memref.store %[[RESULT]], %{{.*}}[%{{.*}}, %{{.*}}]
func.func @basic_addf_accumulation(%i: index, %j: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c210 = arith.constant 210 : index
  %cst = arith.constant 1.0 : f64
  %alloc = memref.alloc() : memref<200x300xf64>
  scf.for %k = %c0 to %c210 step %c1 {
    %val = memref.load %alloc[%i, %j] : memref<200x300xf64>
    %new = arith.addf %val, %cst : f64
    memref.store %new, %alloc[%i, %j] : memref<200x300xf64>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 2: Multi-step chain: load -> mulf -> addf -> store
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @multi_step_chain
// CHECK:         %[[INIT:.*]] = memref.load %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK:         %[[RESULT:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f64)
// CHECK-NOT:       memref.load
// CHECK-NOT:       memref.store
// CHECK:           %[[PRODUCT:.*]] = arith.mulf
// CHECK:           %[[NEW:.*]] = arith.addf %[[ACC]], %[[PRODUCT]]
// CHECK:           scf.yield %[[NEW]]
// CHECK:         memref.store %[[RESULT]], %{{.*}}[%{{.*}}, %{{.*}}]
func.func @multi_step_chain(%i: index, %j: index, %a: f64, %b: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %alloc = memref.alloc() : memref<200x300xf64>
  scf.for %k = %c0 to %c100 step %c1 {
    %val = memref.load %alloc[%i, %j] : memref<200x300xf64>
    %product = arith.mulf %a, %b : f64
    %new = arith.addf %val, %product : f64
    memref.store %new, %alloc[%i, %j] : memref<200x300xf64>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 3: Multiple promotable pairs in same loop
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @multiple_pairs
// CHECK-DAG:     %[[INIT1:.*]] = memref.load %{{.*}}[%{{.*}}]
// CHECK-DAG:     %[[INIT2:.*]] = memref.load %{{.*}}[%{{.*}}]
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC1:.*]] = %[[INIT1]], %[[ACC2:.*]] = %[[INIT2]]) -> (f64, f64)
// CHECK-NOT:       memref.load
// CHECK-NOT:       memref.store
// CHECK-DAG:       arith.addf %[[ACC1]]
// CHECK-DAG:       arith.addf %[[ACC2]]
// CHECK:           scf.yield
// CHECK-DAG:     memref.store %{{.*}}, %{{.*}}[%{{.*}}]
// CHECK-DAG:     memref.store %{{.*}}, %{{.*}}[%{{.*}}]
func.func @multiple_pairs(%i: index, %j: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c50 = arith.constant 50 : index
  %cst1 = arith.constant 1.0 : f64
  %cst2 = arith.constant 2.0 : f64
  %alloc1 = memref.alloc() : memref<100xf64>
  %alloc2 = memref.alloc() : memref<100xf64>
  scf.for %k = %c0 to %c50 step %c1 {
    %val1 = memref.load %alloc1[%i] : memref<100xf64>
    %new1 = arith.addf %val1, %cst1 : f64
    memref.store %new1, %alloc1[%i] : memref<100xf64>
    %val2 = memref.load %alloc2[%j] : memref<100xf64>
    %new2 = arith.addf %val2, %cst2 : f64
    memref.store %new2, %alloc2[%j] : memref<100xf64>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Existing iter_args preserved
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @existing_iter_args
// CHECK:         %[[INIT_LOAD:.*]] = memref.load
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[EXISTING:.*]] = %{{.*}}, %[[ACC:.*]] = %[[INIT_LOAD]]) -> (i32, f64)
// CHECK-NOT:       memref.load
// CHECK-NOT:       memref.store
// CHECK:           arith.addf %[[ACC]]
// CHECK:           scf.yield
// CHECK:         memref.store
func.func @existing_iter_args(%i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.0 : f64
  %c42 = arith.constant 42 : i32
  %alloc = memref.alloc() : memref<100xf64>
  %result = scf.for %k = %c0 to %c10 step %c1 iter_args(%acc = %c42) -> (i32) {
    %val = memref.load %alloc[%i] : memref<100xf64>
    %new = arith.addf %val, %cst : f64
    memref.store %new, %alloc[%i] : memref<100xf64>
    scf.yield %acc : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Integer accumulation (addi)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @integer_accumulation
// CHECK:         %[[INIT:.*]] = memref.load
// CHECK:         %[[RESULT:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32)
// CHECK-NOT:       memref.load
// CHECK-NOT:       memref.store
// CHECK:           arith.addi %[[ACC]]
// CHECK:           scf.yield
// CHECK:         memref.store %[[RESULT]]
func.func @integer_accumulation(%i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c5 = arith.constant 5 : i32
  %alloc = memref.alloc() : memref<100xi32>
  scf.for %k = %c0 to %c20 step %c1 {
    %val = memref.load %alloc[%i] : memref<100xi32>
    %new = arith.addi %val, %c5 : i32
    memref.store %new, %alloc[%i] : memref<100xi32>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 6: Negative — loop-variant indices (skip)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_variant_indices
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.store
func.func @negative_variant_indices() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.0 : f64
  %alloc = memref.alloc() : memref<100xf64>
  scf.for %k = %c0 to %c10 step %c1 {
    %val = memref.load %alloc[%k] : memref<100xf64>
    %new = arith.addf %val, %cst : f64
    memref.store %new, %alloc[%k] : memref<100xf64>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 7: Negative — multiple loads from same memref (skip)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_multiple_loads
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.load
// CHECK:           memref.store
func.func @negative_multiple_loads(%i: index, %j: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %alloc = memref.alloc() : memref<100x100xf64>
  scf.for %k = %c0 to %c10 step %c1 {
    %val1 = memref.load %alloc[%i, %j] : memref<100x100xf64>
    %val2 = memref.load %alloc[%i, %j] : memref<100x100xf64>
    %new = arith.addf %val1, %val2 : f64
    memref.store %new, %alloc[%i, %j] : memref<100x100xf64>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 8: Negative — store value doesn't depend on load (skip)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_no_dependency
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.store
func.func @negative_no_dependency(%i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 42.0 : f64
  %alloc = memref.alloc() : memref<100xf64>
  scf.for %k = %c0 to %c10 step %c1 {
    %val = memref.load %alloc[%i] : memref<100xf64>
    // Store value does NOT depend on load result.
    memref.store %cst, %alloc[%i] : memref<100xf64>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 9: Negative — function argument memref (could alias)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_func_arg_memref
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.store
func.func @negative_func_arg_memref(%alloc: memref<100xf64>, %i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.0 : f64
  scf.for %k = %c0 to %c10 step %c1 {
    %val = memref.load %alloc[%i] : memref<100xf64>
    %new = arith.addf %val, %cst : f64
    memref.store %new, %alloc[%i] : memref<100xf64>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 10: Negative — dynamic bounds (can't prove positive trip count)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_dynamic_bounds
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.store
func.func @negative_dynamic_bounds(%i: index, %lb: index, %ub: index) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f64
  %alloc = memref.alloc() : memref<100xf64>
  scf.for %k = %lb to %ub step %c1 {
    %val = memref.load %alloc[%i] : memref<100xf64>
    %new = arith.addf %val, %cst : f64
    memref.store %new, %alloc[%i] : memref<100xf64>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 11: Negative — mismatched indices between load and store
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_mismatched_indices
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.store
func.func @negative_mismatched_indices(%i: index, %j: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.0 : f64
  %alloc = memref.alloc() : memref<100x100xf64>
  scf.for %k = %c0 to %c10 step %c1 {
    %val = memref.load %alloc[%i, %j] : memref<100x100xf64>
    %new = arith.addf %val, %cst : f64
    memref.store %new, %alloc[%j, %i] : memref<100x100xf64>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 12: Negative — interfering side effect (function call)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_function_call
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.store
func.func private @side_effect() -> ()
func.func @negative_function_call(%i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.0 : f64
  %alloc = memref.alloc() : memref<100xf64>
  scf.for %k = %c0 to %c10 step %c1 {
    %val = memref.load %alloc[%i] : memref<100xf64>
    %new = arith.addf %val, %cst : f64
    func.call @side_effect() : () -> ()
    memref.store %new, %alloc[%i] : memref<100xf64>
  }
  return
}

//===----------------------------------------------------------------------===//
// Test 13: Negative — nested loop (conservative bail-out)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_nested_loop
// CHECK:         scf.for
// CHECK:           memref.load
// CHECK:           memref.store
func.func @negative_nested_loop(%i: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.0 : f64
  %alloc = memref.alloc() : memref<100xf64>
  %alloc2 = memref.alloc() : memref<100xf64>
  scf.for %k = %c0 to %c10 step %c1 {
    %val = memref.load %alloc[%i] : memref<100xf64>
    scf.for %j = %c0 to %c10 step %c1 {
      memref.store %cst, %alloc2[%j] : memref<100xf64>
    }
    %new = arith.addf %val, %cst : f64
    memref.store %new, %alloc[%i] : memref<100xf64>
  }
  return
}
