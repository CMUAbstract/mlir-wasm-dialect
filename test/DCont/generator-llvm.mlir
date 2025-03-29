module {
  //===--------------------------------------------------------------------===//
  // External Declarations
  // We declare these external functions so we can call them.
  //===--------------------------------------------------------------------===//

  // void @print_i32(i32)
  llvm.func @print_i32(i32)
    attributes { llvm.linkage = #llvm.linkage<external> }

  // void @toggle_gpio()
  llvm.func @toggle_gpio()
    attributes { llvm.linkage = #llvm.linkage<external> }

  // The coroutine intrinsics we'll use:
  // token @llvm.coro.id(i32, ptr, ptr, ptr)
  llvm.func @llvm.coro.id(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
    attributes { llvm.linkage = #llvm.linkage<external> }
  // i1 @llvm.coro.alloc(token)
  llvm.func @llvm.coro.alloc(!llvm.token) -> i1
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
  // void @llvm.coro.resume(ptr)
  llvm.func @llvm.coro.resume(!llvm.ptr)
    attributes { llvm.linkage = #llvm.linkage<external> }
  // void @llvm.coro.destroy(ptr)
  llvm.func @llvm.coro.destroy(!llvm.ptr)
    attributes { llvm.linkage = #llvm.linkage<external> }
  // ptr @llvm.coro.promise(ptr, i32, i1)
  llvm.func @llvm.coro.promise(!llvm.ptr, i32, i1) -> !llvm.ptr
    attributes { llvm.linkage = #llvm.linkage<external> }
  // Memory allocation/free functions
  llvm.func @malloc(i32) -> !llvm.ptr
    attributes { llvm.linkage = #llvm.linkage<external> }
  llvm.func @free(!llvm.ptr)
    attributes { llvm.linkage = #llvm.linkage<external> }

  //===--------------------------------------------------------------------===//
  // Coroutine Task: @task
  //  - Iterates over N times
  //  - Uses the promise to store the current iteration index
  //  - Suspends after each iteration to let the caller retrieve the value
  //  - Cleans up when done
  // Returns a pointer to the coroutine frame (!llvm.ptr).
  //===--------------------------------------------------------------------===//
  llvm.func @task() -> !llvm.ptr 
      attributes { "passthrough" = [ "presplitcoroutine" ] } {
    // Maximum number of iterations
    %N = llvm.mlir.constant(1000000 : i32) : i32
    %align = llvm.mlir.constant(4 : i32) : i32
    %one_i32 = llvm.mlir.constant(1 : i32) : i32
    
    // Allocate storage for the promise (i32)
    %promise = llvm.alloca %align x i32 : (i32) -> !llvm.ptr

    // Set up the coroutine
    %c_zero_i32 = llvm.mlir.constant(0 : i32) : i32
    %null_ptr = llvm.mlir.zero : !llvm.ptr
    
    // Create coroutine ID with the promise pointer
    %id = llvm.call @llvm.coro.id(%c_zero_i32, %promise, %null_ptr, %null_ptr)
      : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token

    %false_i1 = llvm.mlir.constant(0 : i1) : i1
    
    %hdl = llvm.call @llvm.coro.begin(%id, %null_ptr)
      : (!llvm.token, !llvm.ptr) -> !llvm.ptr

    %promise_addr = llvm.call @llvm.coro.promise(%hdl, %align, %false_i1) : (!llvm.ptr, i32, i1) -> !llvm.ptr

    // Start the loop with the initial value of 0
    %zero_i32 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^loop(%zero_i32 : i32)
    
  ^loop(%i : i32):
    // Compare i < N
    %cond = llvm.icmp "slt" %i, %N : i32
    llvm.cond_br %cond, ^body, ^cleanup
    
  ^body:
    // Store the current value in the promise
    llvm.store %i, %promise_addr : i32, !llvm.ptr
    
    // Suspend the coroutine and yield to the caller
    %none = llvm.mlir.none : !llvm.token
    %susp = llvm.call @llvm.coro.suspend(%none, %false_i1)
      : (!llvm.token, i1) -> i8
      
    // Switch on the suspend result
    llvm.switch %susp : i8, ^suspend [
      0: ^resume,  // 0 = resumed
      1: ^cleanup  // 1 = destroy
    ]
    
  ^resume:
    // Increment the counter and continue the loop
    %inc = llvm.add %i, %one_i32 : i32
    llvm.br ^loop(%inc : i32)
    
  ^cleanup:
    // Free the coroutine frame memory
    %mem = llvm.call @llvm.coro.free(%id, %hdl)
      : (!llvm.token, !llvm.ptr) -> !llvm.ptr
    
    llvm.br ^suspend
    
  ^suspend:
    // End the coroutine
    %false_i1_2 = llvm.mlir.constant(0 : i1) : i1
    %none_2 = llvm.mlir.none : !llvm.token
    %_unused = llvm.call @llvm.coro.end(%hdl, %false_i1_2, %none_2)
      : (!llvm.ptr, i1, !llvm.token) -> i1
    
    // Return the coroutine handle
    llvm.return %hdl : !llvm.ptr
  }

  //===--------------------------------------------------------------------===//
  // Main Function: @main
  //  - Creates a coroutine
  //  - In a loop until N:
  //     - Access the value from the coroutine's promise
  //     - Print that value using print_i32
  //     - Resume the coroutine to get the next value
  //  - Destroy the coroutine when done
  //===--------------------------------------------------------------------===//
  llvm.func @main() {
    %N = llvm.mlir.constant(1000000 : i32) : i32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    
    // Create the coroutine
    %hdl = llvm.call @task() : () -> !llvm.ptr
    
    // Get the promise address from the coroutine handle
    %align = llvm.mlir.constant(4 : i32) : i32  // Alignment for i32
    %false_i1 = llvm.mlir.constant(0 : i1) : i1  // from_promise = false
    %promise_addr = llvm.call @llvm.coro.promise(%hdl, %align, %false_i1)
      : (!llvm.ptr, i32, i1) -> !llvm.ptr
    
    // Main loop
    llvm.call @toggle_gpio() : () -> ()
    llvm.br ^loop_main(%c0 : i32)
    
  ^loop_main(%i : i32):
    // Check loop condition
    %cond = llvm.icmp "slt" %i, %N : i32
    llvm.cond_br %cond, ^body_main, ^exit_main
    
  ^body_main:
    // Load the current value from the promise
    %value = llvm.load %promise_addr : !llvm.ptr -> i32

    // Print the value using print_i32
    llvm.call @print_i32(%value) : (i32) -> ()
    
    // Resume the coroutine to advance to the next value
    llvm.call @llvm.coro.resume(%hdl) : (!llvm.ptr) -> ()
    
    // Increment loop counter
    %i_next = llvm.add %i, %c1 : i32
    llvm.br ^loop_main(%i_next : i32)
    
  ^exit_main:
    llvm.call @toggle_gpio() : () -> ()
    // Destroy the coroutine when we're done
    llvm.call @llvm.coro.destroy(%hdl) : (!llvm.ptr) -> ()
    
    // Return from main
    llvm.return
  }
}