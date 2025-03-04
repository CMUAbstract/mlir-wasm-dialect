module {
  // Define a global memref to be shared between tasks
  memref.global "private" @global_data : memref<20xi32> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]>

  func.func @task1() {
      // Work on first half of the global memref (indices 0-9)
      %global_memref = memref.get_global @global_data : memref<20xi32>
      %c0_index = arith.constant 0 : index
      %c10_index = arith.constant 10 : index
      %c1_index = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      
      scf.for %i = %c0_index to %c10_index step %c1_index {
        // Load current value
        %val = memref.load %global_memref[%i] : memref<20xi32>
        
        // Double the value
        %doubled = arith.muli %val, %c2_i32 : i32
        
        // Store the doubled value
        memref.store %doubled, %global_memref[%i] : memref<20xi32>
        
        dcont.suspend () : () -> ()
      }
    func.return
  }
  
  func.func @task2() {
      // Work on second half of the global memref (indices 10-19)
      %global_memref = memref.get_global @global_data : memref<20xi32>
      %c10_index = arith.constant 10 : index
      %c20_index = arith.constant 20 : index
      %c1_index = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      
      scf.for %i = %c10_index to %c20_index step %c1_index {
        // Load current value
        %val = memref.load %global_memref[%i] : memref<20xi32>
        
        // Double the value
        %doubled = arith.muli %val, %c2_i32 : i32
        
        // Store the doubled value
        memref.store %doubled, %global_memref[%i] : memref<20xi32>
        
        dcont.suspend () : () -> ()
      }
    func.return
  }
  
  
  // func.func @main
  func.func @main() {
    %task1_handle = dcont.new @task1 : !dcont.cont<"ct">
    %task2_handle = dcont.new @task2 : !dcont.cont<"ct">
    %storage1 = dcont.storage : !dcont.storage<"ct">
    %storage2 = dcont.storage : !dcont.storage<"ct">
    dcont.store %storage1, %task1_handle : !dcont.cont<"ct"> -> !dcont.storage<"ct">
    dcont.store %storage2, %task2_handle : !dcont.cont<"ct"> -> !dcont.storage<"ct">
    
    %c0_index  = arith.constant 0 : index
    %c20_index = arith.constant 20 : index
    %c1_index  = arith.constant 1 : index
    %c2_index  = arith.constant 2 : index
    
    scf.for %i = %c0_index to %c20_index step %c1_index {
      %rem = arith.remui %i, %c2_index : index
      %cond = arith.cmpi eq, %rem, %c0_index : index

      %loaded = scf.if %cond -> (!dcont.cont<"ct">) {
        %loaded = dcont.load %storage1 : !dcont.storage<"ct"> -> !dcont.cont<"ct">
        scf.yield %loaded : !dcont.cont<"ct">
      } else {
        %loaded = dcont.load %storage2 : !dcont.storage<"ct"> -> !dcont.cont<"ct">
        scf.yield %loaded : !dcont.cont<"ct">
      }

      "dcont.resume"(%loaded) 
        ({ ^bb0(%suspended_cont: !dcont.cont<"ct">): 
          scf.if %cond {
            dcont.store %storage1, %suspended_cont : !dcont.cont<"ct"> -> !dcont.storage<"ct">
          } else {
            dcont.store %storage2, %suspended_cont : !dcont.cont<"ct"> -> !dcont.storage<"ct">
          }
          "dcont.suspend_handler_terminator"() : () -> ()
        }) : (!dcont.cont<"ct">) -> ()
    }

    %global_memref = memref.get_global @global_data : memref<20xi32>
    scf.for %i = %c0_index to %c20_index step %c1_index {
      %val = memref.load %global_memref[%i] : memref<20xi32>
      %val_casted = builtin.unrealized_conversion_cast %val : i32 to !ssawasm<integer 32>
      ssawasm.call @print_i32(%val_casted) : (!ssawasm<integer 32>) -> ()
    }
    
    // Return from main
    func.return
  }
}