module {
  //===--------------------------------------------------===//
  // External Declarations
  //===--------------------------------------------------===//

  llvm.func @print_i32(i32)
    attributes { llvm.linkage = #llvm.linkage<external> }

  llvm.func @malloc(i32) -> !llvm.ptr
    attributes { llvm.linkage = #llvm.linkage<external> }

  llvm.func @free(!llvm.ptr)
    attributes { llvm.linkage = #llvm.linkage<external> }

  llvm.func @llvm.coro.id(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
    attributes { llvm.linkage = #llvm.linkage<external> }

  llvm.func @llvm.coro.size.i32() -> i32
    attributes { llvm.linkage = #llvm.linkage<external> }

  llvm.func @llvm.coro.begin(!llvm.token, !llvm.ptr) -> !llvm.ptr
    attributes { llvm.linkage = #llvm.linkage<external> }

  llvm.func @llvm.coro.suspend(!llvm.token, i1) -> i8
    attributes { llvm.linkage = #llvm.linkage<external> }

  llvm.func @llvm.coro.free(!llvm.token, !llvm.ptr) -> !llvm.ptr
    attributes { llvm.linkage = #llvm.linkage<external> }

  llvm.func @llvm.coro.end(!llvm.ptr, i1, !llvm.token) -> i1
    attributes { llvm.linkage = #llvm.linkage<external> }

  llvm.func @llvm.coro.resume(!llvm.ptr)
    attributes { llvm.linkage = #llvm.linkage<external> }

  //===--------------------------------------------------===//
  // Global array: [20 x i32]
  // Lowered from memref<20xi32> = dense<...>
  //===--------------------------------------------------===//
  llvm.mlir.global private @global_data(dense<[
    1, 2, 3, 4, 5, 
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20
  ]>  : tensor<20xi32>) : !llvm.array<20 x i32>

  //===--------------------------------------------------===//
  // Task1: Loops i in [0..10), doubling global_data[i]
  //        Suspends each iteration
  //===--------------------------------------------------===//
  llvm.func @task1() -> !llvm.ptr
      attributes { "passthrough" = [ "presplitcoroutine" ] } {
    %c10 = llvm.mlir.constant(10 : i32) : i32
    %c2  = llvm.mlir.constant(2 : i32) : i32
    llvm.br ^entry

  ^entry:
    %c0_i32  = llvm.mlir.constant(0 : i32) : i32
    %null    = llvm.mlir.zero : !llvm.ptr
    %id   = llvm.call @llvm.coro.id(%c0_i32, %null, %null, %null)
             : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
    %sz   = llvm.call @llvm.coro.size.i32() : () -> i32
    %mem  = llvm.call @malloc(%sz) : (i32) -> !llvm.ptr
    %hdl  = llvm.call @llvm.coro.begin(%id, %mem)
             : (!llvm.token, !llvm.ptr) -> !llvm.ptr

    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^loop(%zero : i32)

  ^loop(%i : i32):
    %cond = llvm.icmp "slt" %i, %c10 : i32
    llvm.cond_br %cond, ^body, ^cleanup

  ^body:
    %gbase = llvm.mlir.addressof @global_data : !llvm.ptr
    %ptr   = llvm.getelementptr %gbase[%i]
               : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %val   = llvm.load %ptr : !llvm.ptr -> i32
    %dbl   = llvm.mul %val, %c2 : i32
    llvm.store %dbl, %ptr : i32, !llvm.ptr

    %false = llvm.mlir.constant(0 : i1) : i1
    %none  = llvm.mlir.none : !llvm.token
    %susp  = llvm.call @llvm.coro.suspend(%none, %false)
               : (!llvm.token, i1) -> i8
    llvm.switch %susp : i8, ^sw_default [
      0: ^resume,
      1: ^cleanup
    ]

  ^sw_default:
    llvm.br ^cleanup

  ^resume:
    %one = llvm.mlir.constant(1 : i32) : i32
    %inc = llvm.add %i, %one : i32
    llvm.br ^loop(%inc : i32)

  ^cleanup:
    %mem2 = llvm.call @llvm.coro.free(%id, %hdl) : (!llvm.token, !llvm.ptr) -> !llvm.ptr
    llvm.call @free(%mem2) : (!llvm.ptr) -> ()
    llvm.br ^suspend

  ^suspend:
    %f2 = llvm.mlir.constant(0 : i1) : i1
    %n2 = llvm.mlir.none : !llvm.token
    %_unused = llvm.call @llvm.coro.end(%hdl, %f2, %n2)
                : (!llvm.ptr, i1, !llvm.token) -> i1
    llvm.return %hdl : !llvm.ptr
  }

  //===--------------------------------------------------===//
  // Task2: Loops i in [10..20), doubling global_data[i]
  //        Suspends each iteration
  //===--------------------------------------------------===//
  llvm.func @task2() -> !llvm.ptr
      attributes { "passthrough" = [ "presplitcoroutine" ] } {
    %c10 = llvm.mlir.constant(10 : i32) : i32
    %c20 = llvm.mlir.constant(20 : i32) : i32
    %c2  = llvm.mlir.constant(2 : i32) : i32
    llvm.br ^entry

  ^entry:
    %c0_i32  = llvm.mlir.constant(0 : i32) : i32
    %null    = llvm.mlir.zero : !llvm.ptr
    %id   = llvm.call @llvm.coro.id(%c0_i32, %null, %null, %null)
             : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
    %sz   = llvm.call @llvm.coro.size.i32() : () -> i32
    %mem  = llvm.call @malloc(%sz) : (i32) -> !llvm.ptr
    %hdl  = llvm.call @llvm.coro.begin(%id, %mem)
             : (!llvm.token, !llvm.ptr) -> !llvm.ptr

    llvm.br ^loop(%c10 : i32)

  ^loop(%i : i32):
    %cond = llvm.icmp "slt" %i, %c20 : i32
    llvm.cond_br %cond, ^body, ^cleanup

  ^body:
    %gbase = llvm.mlir.addressof @global_data : !llvm.ptr
    %ptr   = llvm.getelementptr %gbase[%i]
               : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %val   = llvm.load %ptr : !llvm.ptr -> i32
    %dbl   = llvm.mul %val, %c2 : i32
    llvm.store %dbl, %ptr : i32, !llvm.ptr

    %false = llvm.mlir.constant(0 : i1) : i1
    %none  = llvm.mlir.none : !llvm.token
    %susp  = llvm.call @llvm.coro.suspend(%none, %false)
               : (!llvm.token, i1) -> i8
    llvm.switch %susp : i8, ^sw_default [
      0: ^resume,
      1: ^cleanup
    ]

  ^sw_default:
    llvm.br ^cleanup

  ^resume:
    %one = llvm.mlir.constant(1 : i32) : i32
    %inc = llvm.add %i, %one : i32
    llvm.br ^loop(%inc : i32)

  ^cleanup:
    %mem2 = llvm.call @llvm.coro.free(%id, %hdl)
               : (!llvm.token, !llvm.ptr) -> !llvm.ptr
    llvm.call @free(%mem2) : (!llvm.ptr) -> ()
    llvm.br ^suspend

  ^suspend:
    %f2 = llvm.mlir.constant(0 : i1) : i1
    %n2 = llvm.mlir.none : !llvm.token
    %_unused = llvm.call @llvm.coro.end(%hdl, %f2, %n2)
               : (!llvm.ptr, i1, !llvm.token) -> i1
    llvm.return %hdl : !llvm.ptr
  }

  //===--------------------------------------------------===//
  // main: 
  //   - create the two coroutine handles
  //   - for i in [0..20):
  //       if (i % 2 == 0) resume task1 else task2
  //   - print final array
  //===--------------------------------------------------===//
  llvm.func @main() {
    %c0  = llvm.mlir.constant(0 : i32) : i32
    %c1  = llvm.mlir.constant(1 : i32) : i32
    %c2  = llvm.mlir.constant(2 : i32) : i32
    %c20 = llvm.mlir.constant(20 : i32) : i32

    // Create coroutines
    %hdl1 = llvm.call @task1() : () -> !llvm.ptr
    %hdl2 = llvm.call @task2() : () -> !llvm.ptr

    // Loop i=0..20
    llvm.br ^loop_main(%c0 : i32)

  ^loop_main(%i : i32):
    %cond = llvm.icmp "slt" %i, %c20 : i32
    llvm.cond_br %cond, ^body_main, ^exit_main

  ^body_main:
    // i % 2
    %r = llvm.urem %i, %c2 : i32
    // Compare to zero
    %eq_zero = llvm.icmp "eq" %r, %c0 : i32
    llvm.cond_br %eq_zero, ^resume1, ^resume2

  ^resume1:
    llvm.call @llvm.coro.resume(%hdl1) : (!llvm.ptr) -> ()
    llvm.br ^inc_i

  ^resume2:
    llvm.call @llvm.coro.resume(%hdl2) : (!llvm.ptr) -> ()
    llvm.br ^inc_i

  ^inc_i:
    %i2 = llvm.add %i, %c1 : i32
    llvm.br ^loop_main(%i2 : i32)

  ^exit_main:
    // print final array contents
    llvm.br ^loop_print(%c0 : i32)

  ^loop_print(%ip : i32):
    %cond2 = llvm.icmp "slt" %ip, %c20 : i32
    llvm.cond_br %cond2, ^print_body, ^done

  ^print_body:
    %gbase = llvm.mlir.addressof @global_data : !llvm.ptr
    %elem_ptr = llvm.getelementptr %gbase[%ip]
                  : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %val = llvm.load %elem_ptr : !llvm.ptr -> i32
    llvm.call @print_i32(%val) : (i32) -> ()

    %ip1 = llvm.add %ip, %c1 : i32
    llvm.br ^loop_print(%ip1 : i32)

  ^done:
    llvm.return
  }
}
