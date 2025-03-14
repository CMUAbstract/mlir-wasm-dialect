module {
  func.func private @print_i32(i32) -> ()

  async.func @task1() -> !async.token {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32

    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %c1000_index = arith.constant 1000 : index

    scf.for %i = %c0_index to %c1000_index step %c1_index {
      func.call @print_i32(%c1) : (i32) -> ()

      %token = call @task2() : () -> !async.token
      async.await %token : !async.token

      func.call @print_i32(%c2) : (i32) -> ()
    }


    async.return
  }

  async.func @task2() -> !async.token {
    %c3 = arith.constant 3 : i32
    func.call @print_i32(%c3) : (i32) -> ()

    async.return
  }

  func.func @main() {
    %token = async.call @task1() : () -> !async.token

    async.await %token : !async.token 

    return 
  }
}