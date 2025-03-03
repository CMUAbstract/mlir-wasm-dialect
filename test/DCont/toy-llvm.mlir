module {
  //===--------------------------------------------------------------------===//
  // External Declarations
  // We declare these external functions so we can call them.
  //===--------------------------------------------------------------------===//

  // void @print_i32(i32)
  llvm.func @print_i32(i32)
    attributes { llvm.linkage = #llvm.linkage<external> }

  // void* @malloc(i32)
  llvm.func @malloc(i32) -> !llvm.ptr
    attributes { llvm.linkage = #llvm.linkage<external> }

  // void @free(void*)
  llvm.func @free(!llvm.ptr)
    attributes { llvm.linkage = #llvm.linkage<external> }

  // The coroutine intrinsics we’ll use:
  // token @llvm.coro.id(i32, ptr, ptr, ptr)
  llvm.func @llvm.coro.id(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
    attributes { llvm.linkage = #llvm.linkage<external> }
  // i32 @llvm.coro.size.i32()
  llvm.func @llvm.coro.size.i32() -> i32
    attributes { llvm.linkage = #llvm.linkage<external> }
  // void* @llvm.coro.begin(token, void*)
  llvm.func @llvm.coro.begin(!llvm.token, !llvm.ptr) -> !llvm.ptr
    attributes { llvm.linkage = #llvm.linkage<external> }
  // i8 @llvm.coro.suspend(token, i1)
  llvm.func @llvm.coro.suspend(!llvm.token, i1) -> i8
    attributes { llvm.linkage = #llvm.linkage<external> }
  // void* @llvm.coro.free(token, void*)
  llvm.func @llvm.coro.free(!llvm.token, !llvm.ptr) -> !llvm.ptr
    attributes { llvm.linkage = #llvm.linkage<external> }
  // i1 @llvm.coro.end(void*, i1, token)
  llvm.func @llvm.coro.end(!llvm.ptr, i1, !llvm.token) -> i1
    attributes { llvm.linkage = #llvm.linkage<external> }
  llvm.func @llvm.coro.resume(!llvm.ptr)
  //===--------------------------------------------------------------------===//
  // Coroutine Task: @task
  //  - Iterates over N times
  //  - Prints iteration index
  //  - Suspends after each iteration
  //  - Cleans up when done
  // Returns a pointer to the coroutine frame (!llvm.ptr).
  //===--------------------------------------------------------------------===//
  llvm.func @task() -> !llvm.ptr
      attributes { "passthrough" = [ "presplitcoroutine" ] } {
    // We'll define 4 blocks:
    // ^entry, ^loop, ^cleanup, ^suspend

    %N = llvm.mlir.constant(1000 : i32) : i32

    // ^entry
    llvm.br ^entry_block

  ^entry_block:  // set up coroutine, then jump to ^loop
    // 1) coro.id
    %c_zero_i32 = llvm.mlir.constant(0 : i32) : i32
    %null_ptr = llvm.mlir.zero : !llvm.ptr
    %id = llvm.call @llvm.coro.id(%c_zero_i32, %null_ptr, %null_ptr, %null_ptr)
      : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
    // 2) coro.size.i32
    %sz = llvm.call @llvm.coro.size.i32() : () -> i32
    // 3) malloc
    %mem = llvm.call @malloc(%sz) : (i32) -> !llvm.ptr
    // 4) coro.begin
    %hdl = llvm.call @llvm.coro.begin(%id, %mem)
      : (!llvm.token, !llvm.ptr) -> !llvm.ptr

    // We'll start iteration count at 0
    %zero_i32 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^loop(%zero_i32 : i32)

  // ^loop takes arguments: (hdl, i)
  ^loop(%i : i32):
    // Compare i < N
    %cond = llvm.icmp "slt" %i, %N : i32
    llvm.cond_br %cond, ^body, ^cleanup

  // ^body is the body of the loop: print_i32 i, suspend, increment i, back to loop
  ^body:
    // call @print_i32(i)
    llvm.call @print_i32(%i) : (i32) -> ()

    // Perform a suspend
    %none = llvm.mlir.none : !llvm.token
    %false_i1 = llvm.mlir.constant(0 : i1) : i1
    %susp = llvm.call @llvm.coro.suspend(%none, %false_i1)
      : (!llvm.token, i1) -> i8

    // Switch on the result
    //   0 => resumed => continue loop
    //   1 => cleanup
    // We'll do this with an llvm.switch.
    // First define a block for "resume" and one for "cleanup".
    llvm.switch %susp : i8, ^suspend [
      0: ^resume,
      1: ^cleanup
    ]

  // ^resume: increment i, branch to ^loop
  ^resume:
    %one_i32 = llvm.mlir.constant(1 : i32) : i32
    %inc = llvm.add %i, %one_i32 : i32
    llvm.br ^loop(%inc : i32)

  // ^cleanup(hdl) => block argument for handle
  ^cleanup:
    // 1) call coro.free
    %mem2 = llvm.call @llvm.coro.free(%id, %hdl)
      : (!llvm.token, !llvm.ptr) -> !llvm.ptr
    // 2) free
    llvm.call @free(%mem2) : (!llvm.ptr) -> ()
    // branch to ^suspend
    llvm.br ^suspend

  // ^suspend(hdl)
  ^suspend:
    %false_i1_2 = llvm.mlir.constant(0 : i1) : i1
    %none_2 = llvm.mlir.none : !llvm.token
    %_unused = llvm.call @llvm.coro.end(%hdl, %false_i1_2, %none_2)
      : (!llvm.ptr, i1, !llvm.token) -> i1
    // Return the coroutine handle
    llvm.return %hdl : !llvm.ptr
  }

  //===--------------------------------------------------------------------===//
  // Main Function: @main
  //  - for i in [0..N):
  //       call @task()
  //       call @print_i32(0)
  //  - return 0
  //===--------------------------------------------------------------------===//
  llvm.func @main() {
    // We'll define N=3 again
    %N = llvm.mlir.constant(1000 : i32) : i32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32

    %hdl = llvm.call @task() : () -> !llvm.ptr

    // Jump to ^loop_main with i=0
    llvm.br ^loop_main(%c0 : i32)

  ^loop_main(%i : i32):
    // i < 3?
    %cond = llvm.icmp "slt" %i, %N : i32
    llvm.cond_br %cond, ^body_main, ^exit_main

  ^body_main:
    // 1) call @task()
    // (We don't actually need %hdl for anything else here)
    llvm.call @llvm.coro.resume(%hdl) : (!llvm.ptr) -> ()

    // 2) call @print_i32(0)
    llvm.call @print_i32(%c0) : (i32) -> ()

    // increment i
    %i2 = llvm.add %i, %c1 : i32
    llvm.br ^loop_main(%i2 : i32)

  ^exit_main:
    // return 0
    llvm.return 
  }
}
