module {

  func.func @task1() {
      %c0_index = arith.constant 0 : index
      %c10_index = arith.constant 1000000 : index
      %c1_index = arith.constant 1 : index

      scf.for %i = %c0_index to %c10_index step %c1_index {
        %i_i32 = arith.index_cast %i : index to i32
        dcont.suspend (%i_i32) : (i32) -> ()
      }
    func.return
  }


  // func.func @main
  func.func @main() {
    %task1_handle = dcont.new @task1 : !dcont.cont<(i32)->()>

    %storage = dcont.storage : !dcont.storage<(i32)->()>
    dcont.store %storage, %task1_handle : !dcont.cont<(i32)->()> -> !dcont.storage<(i32)->()>

    %c0_index  = arith.constant 0 : index
    %c10_index = arith.constant 1000000 : index
    %c1_index  = arith.constant 1 : index

    ssawasm.call @toggle_gpio() : () -> ()
    scf.for %i = %c0_index to %c10_index step %c1_index {
      %loaded = dcont.load %storage : !dcont.storage<(i32)->()> -> !dcont.cont<(i32)->()>
      "dcont.resume"(%loaded) 
        ({ ^bb0(%i_i32: i32, %suspended_cont: !dcont.cont<(i32)->()>): 
          ssawasm.call @print_i32(%i_i32) : (i32) -> ()
        
          dcont.store %storage, %suspended_cont : !dcont.cont<(i32)->()> -> !dcont.storage<(i32)->()>
          dcont.suspend_handler_return
        }) : (!dcont.cont<(i32)->()>) -> ()
    }
    ssawasm.call @toggle_gpio() : () -> ()

    // Return from main
    func.return
  }
}
