module {
  func.func @main() -> i32 attributes { exported } {
    %c2800 = arith.constant 2800 : index
    %c7_i32 = arith.constant 7 : i32
    %c1_i32 = arith.constant 1 : i32
    %c13_i32 = arith.constant 13 : i32
    %c11_i32 = arith.constant 11 : i32
    %c999_i32 = arith.constant 999 : i32
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<2800x2800xi32>
    scf.for %arg0 = %c0 to %c2800 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c2800 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.muli %0, %1 : i32
        %3 = arith.remsi %2, %c7_i32 : i32
        %4 = arith.addi %3, %c1_i32 : i32
        memref.store %4, %alloc[%arg0, %arg1] : memref<2800x2800xi32>
        %5 = arith.addi %0, %1 : i32
        %6 = arith.remsi %5, %c13_i32 : i32
        %7 = arith.cmpi eq, %6, %c0_i32 : i32
        %8 = scf.if %7 -> (i1) {
          scf.yield %true : i1
        } else {
          %10 = arith.remsi %5, %c7_i32 : i32
          %11 = arith.cmpi eq, %10, %c0_i32 : i32
          scf.yield %11 : i1
        }
        %9 = scf.if %8 -> (i1) {
          scf.yield %true : i1
        } else {
          %10 = arith.remsi %5, %c11_i32 : i32
          %11 = arith.cmpi eq, %10, %c0_i32 : i32
          scf.yield %11 : i1
        }
        scf.if %9 {
          memref.store %c999_i32, %alloc[%arg0, %arg1] : memref<2800x2800xi32>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 2800 {
      affine.for %arg1 = 0 to 2800 {
        affine.for %arg2 = 0 to 2800 {
          %0 = affine.load %alloc[%arg1, %arg2] : memref<2800x2800xi32>
          %1 = affine.load %alloc[%arg1, %arg0] : memref<2800x2800xi32>
          %2 = affine.load %alloc[%arg0, %arg2] : memref<2800x2800xi32>
          %3 = arith.addi %1, %2 : i32
          %4 = arith.cmpi slt, %0, %3 : i32
          %5 = arith.select %4, %0, %3 : i32
          affine.store %5, %alloc[%arg1, %arg2] : memref<2800x2800xi32>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c2800 step %c1 {
      scf.for %arg1 = %c0 to %c2800 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<2800x2800xi32>
        func.call @print_i32(%0) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<2800x2800xi32>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
