module {
  func.func @main() -> i32 {
    %c1 = arith.constant 1 : index
    %c11 = arith.constant 11 : index  // upper bound is exclusive, so go up to 11
    %zero_i32 = arith.constant 0 : i32
    %num = arith.constant 5 : i32

    %sum = scf.for %i = %c1 to %c11 step %c1 iter_args(%acc = %zero_i32) -> (i32) {
      
      %new_acc = arith.addi %acc, %num : i32
      
      scf.yield %new_acc : i32
    }

    // Return the result
    func.return %sum : i32
  }
}
