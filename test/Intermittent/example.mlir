module {
  memref.global @global_var_2 : memref<1xi32> = dense<0> {intermittent.nonvolatile}
  memref.global @global_var_1 : memref<1xi32> = dense<0> {intermittent.nonvolatile}
  func.func @task_init() -> i32 attributes {intermittent.task = 0 : i32} {
    %c1_i32 = arith.constant 1 : i32
    return %c1_i32 : i32
  }
  func.func @task_a() -> i32 attributes {intermittent.task = 1 : i32} {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = memref.get_global @global_var_1 : memref<1xi32>
    %1 = affine.load %0[0] : memref<1xi32>
    %2 = arith.addi %1, %c1_i32 : i32
    affine.store %2, %0[0] : memref<1xi32>
    return %c2_i32 : i32
  }
  func.func @task_b() -> i32 attributes {intermittent.task = 2 : i32} {
    %c1_i32 = arith.constant 1 : i32
    %0 = memref.get_global @global_var_2 : memref<1xi32>
    %1 = affine.load %0[0] : memref<1xi32>
    %2 = arith.addi %1, %c1_i32 : i32
    affine.store %2, %0[0] : memref<1xi32>
    return %c1_i32 : i32
  }
}
