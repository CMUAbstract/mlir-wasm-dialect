module {
  func.func @main() -> i32 attributes { exported } {
    %c400 = arith.constant 400 : index
    %cst = arith.constant 4.000000e+02 : f64
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c400_i32 = arith.constant 400 : i32
    %alloc = memref.alloc() : memref<400xf64>
    %alloc_0 = memref.alloc() : memref<400xf64>
    scf.for %arg0 = %c0 to %c400 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.remsi %0, %c400_i32 : i32
      %2 = arith.sitofp %1 : i32 to f64
      %3 = arith.divf %2, %cst : f64
      memref.store %3, %alloc[%arg0] : memref<400xf64>
      %4 = arith.addi %0, %c1_i32 : i32
      %5 = arith.remsi %4, %c400_i32 : i32
      %6 = arith.sitofp %5 : i32 to f64
      %7 = arith.divf %6, %cst : f64
      memref.store %7, %alloc_0[%arg0] : memref<400xf64>
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 400 {
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 400 {
        %1 = affine.load %alloc[%arg0] : memref<400xf64>
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.muli %0, %2 : i32
        %4 = arith.remsi %3, %c400_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        %7 = arith.addi %2, %c3_i32 : i32
        %8 = arith.remsi %7, %c400_i32 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst : f64
        %11 = arith.mulf %6, %10 : f64
        %12 = arith.addf %1, %11 : f64
        affine.store %12, %alloc[%arg0] : memref<400xf64>
      }
    }
    affine.for %arg0 = 0 to 400 {
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 400 {
        %1 = affine.load %alloc_0[%arg0] : memref<400xf64>
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.muli %2, %0 : i32
        %4 = arith.remsi %3, %c400_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        %7 = arith.addi %2, %c4_i32 : i32
        %8 = arith.remsi %7, %c400_i32 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst : f64
        %11 = arith.mulf %6, %10 : f64
        %12 = arith.addf %1, %11 : f64
        affine.store %12, %alloc_0[%arg0] : memref<400xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c400 step %c1 {
      %0 = memref.load %alloc[%arg0] : memref<400xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    scf.for %arg0 = %c0 to %c400 step %c1 {
      %0 = memref.load %alloc_0[%arg0] : memref<400xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<400xf64>
    memref.dealloc %alloc_0 : memref<400xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
