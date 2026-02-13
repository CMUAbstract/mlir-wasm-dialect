module {
  func.func @main() -> i32 attributes { exported } {
    %c120 = arith.constant 120 : index
    %cst = arith.constant 1.200000e+02 : f64
    %cst_0 = arith.constant 9.000000e+00 : f64
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 2.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<120x120xf64>
    scf.for %arg0 = %c0 to %c120 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.sitofp %0 : i32 to f64
      scf.for %arg1 = %c0 to %c120 step %c1 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.addi %2, %c2_i32 : i32
        %4 = arith.sitofp %3 : i32 to f64
        %5 = arith.mulf %1, %4 : f64
        %6 = arith.addf %5, %cst_1 : f64
        %7 = arith.divf %6, %cst : f64
        memref.store %7, %alloc[%arg0, %arg1] : memref<120x120xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 40 {
      affine.for %arg1 = 1 to 119 {
        affine.for %arg2 = 1 to 119 {
          %0 = affine.load %alloc[%arg1 - 1, %arg2 - 1] : memref<120x120xf64>
          %1 = affine.load %alloc[%arg1 - 1, %arg2] : memref<120x120xf64>
          %2 = arith.addf %0, %1 : f64
          %3 = affine.load %alloc[%arg1 - 1, %arg2 + 1] : memref<120x120xf64>
          %4 = arith.addf %2, %3 : f64
          %5 = affine.load %alloc[%arg1, %arg2 - 1] : memref<120x120xf64>
          %6 = arith.addf %4, %5 : f64
          %7 = affine.load %alloc[%arg1, %arg2] : memref<120x120xf64>
          %8 = arith.addf %6, %7 : f64
          %9 = affine.load %alloc[%arg1, %arg2 + 1] : memref<120x120xf64>
          %10 = arith.addf %8, %9 : f64
          %11 = affine.load %alloc[%arg1 + 1, %arg2 - 1] : memref<120x120xf64>
          %12 = arith.addf %10, %11 : f64
          %13 = affine.load %alloc[%arg1 + 1, %arg2] : memref<120x120xf64>
          %14 = arith.addf %12, %13 : f64
          %15 = affine.load %alloc[%arg1 + 1, %arg2 + 1] : memref<120x120xf64>
          %16 = arith.addf %14, %15 : f64
          %17 = arith.divf %16, %cst_0 : f64
          affine.store %17, %alloc[%arg1, %arg2] : memref<120x120xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c120 step %c1 {
      scf.for %arg1 = %c0 to %c120 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<120x120xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<120x120xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
