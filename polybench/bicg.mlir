module {
  func.func @main() -> i32 {
    %c124 = arith.constant 124 : index
    %c116 = arith.constant 116 : index
    %cst = arith.constant 1.240000e+02 : f64
    %cst_0 = arith.constant 1.160000e+02 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c116_i32 = arith.constant 116 : i32
    %c124_i32 = arith.constant 124 : i32
    %alloc = memref.alloc() : memref<116xf64>
    %alloc_2 = memref.alloc() : memref<124xf64>
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 116 {
      affine.store %cst_1, %alloc[%arg0] : memref<116xf64>
    }
    affine.for %arg0 = 0 to 124 {
      affine.store %cst_1, %alloc_2[%arg0] : memref<124xf64>
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.remsi %0, %c124_i32 : i32
      %2 = arith.sitofp %1 : i32 to f64
      %3 = arith.divf %2, %cst : f64
      affine.for %arg1 = 0 to 116 {
        %4 = affine.load %alloc[%arg1] : memref<116xf64>
        %5 = arith.index_cast %arg1 : index to i32
        %6 = arith.addi %5, %c1_i32 : i32
        %7 = arith.muli %0, %6 : i32
        %8 = arith.remsi %7, %c124_i32 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst : f64
        %11 = arith.mulf %3, %10 : f64
        %12 = arith.addf %4, %11 : f64
        affine.store %12, %alloc[%arg1] : memref<116xf64>
        %13 = affine.load %alloc_2[%arg0] : memref<124xf64>
        %14 = arith.remsi %5, %c116_i32 : i32
        %15 = arith.sitofp %14 : i32 to f64
        %16 = arith.divf %15, %cst_0 : f64
        %17 = arith.mulf %10, %16 : f64
        %18 = arith.addf %13, %17 : f64
        affine.store %18, %alloc_2[%arg0] : memref<124xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c116 step %c1 {
      %0 = memref.load %alloc[%arg0] : memref<116xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    scf.for %arg0 = %c0 to %c124 step %c1 {
      %0 = memref.load %alloc_2[%arg0] : memref<124xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<116xf64>
    memref.dealloc %alloc_2 : memref<124xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
