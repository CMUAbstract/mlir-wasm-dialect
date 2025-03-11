module {

  func.func @task1() {
      %c0_index = arith.constant 0 : index
      %c10_index = arith.constant 1000 : index
      %c1_index = arith.constant 1 : index

      scf.for %i = %c0_index to %c10_index step %c1_index {
        %i_i32 = arith.index_cast %i : index to i32
        %i_i32_2 = builtin.unrealized_conversion_cast %i_i32 : i32 to !ssawasm<integer 32>
        ssawasm.call @print_i32(%i_i32_2) : (!ssawasm<integer 32>) -> ()

        dcont.suspend () : () -> ()
      }
    func.return
  }


  // func.func @main
  func.func @main() {
    %task1_handle = dcont.new @task1 : !dcont.cont<"ct">

    %storage = dcont.storage : !dcont.storage<"ct">
    dcont.store %storage, %task1_handle : !dcont.cont<"ct"> -> !dcont.storage<"ct">

    %c0_index  = arith.constant 0 : index
    %c10_index = arith.constant 1000 : index
    %c1_index  = arith.constant 1 : index

    scf.for %i = %c0_index to %c10_index step %c1_index {
      %loaded = dcont.load %storage : !dcont.storage<"ct"> -> !dcont.cont<"ct">
      "dcont.resume"(%loaded) 
        ({ ^bb0(%suspended_cont: !dcont.cont<"ct">): 
          dcont.store %storage, %suspended_cont : !dcont.cont<"ct"> -> !dcont.storage<"ct">
          dcont.suspend_handler_return : () -> ()
        }) : (!dcont.cont<"ct">) -> ()
      %x = ssawasm.constant 0 : i32 !ssawasm<integer 32>
      ssawasm.call @print_i32(%x) : (!ssawasm<integer 32>) -> ()
    }

    // Return from main
    func.return
  }
}
