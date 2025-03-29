module {
 func.func @main() {
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c10 = arith.constant 10 : index
   scf.for %iv = %c0 to %c10 step %c1 {
      %t = arith.addi %iv, %iv : index
   }
   return
 }

}
