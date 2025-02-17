module {
  // Global memref @a initialized to 0 (rank-0 memref of i32)
  memref.global "private" @a : memref<i32> = dense<0> 

  // Global memref @b initialized to 100 (rank-0 memref of i32)
  memref.global "private" @b : memref<i32> = dense<100> 

  // func.func @task1
  func.func @task1() {
    // Obtain memref for @a
    %0 = memref.get_global @a : memref<i32>

    // Load the integer value in @a
    %a_value = memref.load %0[] : memref<i32>

    %c0_index = arith.constant 0 : index
    %c100_index = arith.constant 100 : index
    %c1_index = arith.constant 1 : index
    %c1_i32   = arith.constant 1 : i32

    // scf.for i = 0 to the current value in @a
    scf.for %i = %c0_index to %c100_index step %c1_index {
      // Print or do something with the current @a value
      %curr_a = memref.load %0[] : memref<i32>
      ssawasm.call @print_i32(%curr_a) : (i32) -> ()

      // increment @a by 1
      %incremented = arith.addi %curr_a, %c1_i32 : i32
      memref.store %incremented, %0[] : memref<i32>

      // Suspend continuation
      dcont.suspend () : () -> ()
    }

    // Return from task1
    func.return
  }

  // func.func @task2
  func.func @task2() {
    // Obtain memref for @b
    %0 = memref.get_global @b : memref<i32>

    %c0_index  = arith.constant 0 : index
    %c100_index = arith.constant 100 : index
    %c1_index  = arith.constant 1 : index
    %c1_i32    = arith.constant 1 : i32

    // scf.for i = 0 to 100
    scf.for %i = %c0_index to %c100_index step %c1_index {
      // read @b
      %curr_b = memref.load %0[] : memref<i32>
      ssawasm.call @print_i32(%curr_b) : (i32) -> ()
      // increment @b by 1
      %incremented = arith.addi %curr_b, %c1_i32 : i32
      memref.store %incremented, %0[] : memref<i32>

      // Suspend continuation
      dcont.suspend () : () -> ()
    }

    // Return from task2
    func.return
  }

  // func.func @main
  func.func @main() {
    // Create new delimited continuations for task1 and task2
    %task1_handle = dcont.new @task1 : !dcont.cont<"ct">
    %task2_handle = dcont.new @task2 : !dcont.cont<"ct">

    %c0_index  = arith.constant 0 : index
    %c100_index = arith.constant 100 : index
    %c1_index  = arith.constant 1 : index

    %c2_i32 = arith.constant 2 : i32 %c0_i32 = arith.constant 0 : i32

    // scf.for i = 0 to 100
    scf.for %i = %c0_index to %c100_index step %c1_index {
      // Convert loop index from index to i32
      %i_i32 = arith.index_cast %i : index to i32

      // i % 2
      %mod = arith.remui %i_i32, %c2_i32 : i32

      // check if (i % 2 == 0)
      %cond = arith.cmpi eq, %mod, %c0_i32 : i32

      // if true, resume task1; else, resume task2
      %task_handle = arith.select %cond, %task1_handle, %task2_handle : !dcont.cont<"ct">
      %out = dcont.resume (%task_handle : !dcont.cont<"ct">) : () -> (), !dcont.cont<"ct">
    }

    // Return from main
    func.return
  }
}
