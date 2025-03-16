module {
  func.func @main() -> i32 {
    %c124 = arith.constant 124 : index
    %cst = arith.constant 5.800000e+02 : f64
    %cst_0 = arith.constant 1.240000e+02 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c124_i32 = arith.constant 124 : i32
    %alloc = memref.alloc() : memref<124xf64>
    %alloc_3 = memref.alloc() : memref<116xf64>
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 124 {
      affine.store %cst_1, %alloc[%arg0] : memref<124xf64>
    }
    affine.for %arg0 = 0 to 116 {
      affine.store %cst_1, %alloc_3[%arg0] : memref<116xf64>
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 124 {
        %3 = affine.load %alloc_3[%arg0] : memref<116xf64>
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.addi %0, %4 : i32
        %6 = arith.remsi %5, %c124_i32 : i32
        %7 = arith.sitofp %6 : i32 to f64
        %8 = arith.divf %7, %cst : f64
        %9 = arith.sitofp %4 : i32 to f64
        %10 = arith.divf %9, %cst_0 : f64
        %11 = arith.addf %10, %cst_2 : f64
        %12 = arith.mulf %8, %11 : f64
        %13 = arith.addf %3, %12 : f64
        affine.store %13, %alloc_3[%arg0] : memref<116xf64>
      }
      %1 = arith.index_cast %arg0 : index to i32
      %2 = affine.load %alloc_3[%arg0] : memref<116xf64>
      affine.for %arg1 = 0 to 124 {
        %3 = affine.load %alloc[%arg1] : memref<124xf64>
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.addi %1, %4 : i32
        %6 = arith.remsi %5, %c124_i32 : i32
        %7 = arith.sitofp %6 : i32 to f64
        %8 = arith.divf %7, %cst : f64
        %9 = arith.mulf %8, %2 : f64
        %10 = arith.addf %3, %9 : f64
        affine.store %10, %alloc[%arg1] : memref<124xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c124 step %c1 {
      %0 = memref.load %alloc[%arg0] : memref<124xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<124xf64>
    memref.dealloc %alloc_3 : memref<116xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
