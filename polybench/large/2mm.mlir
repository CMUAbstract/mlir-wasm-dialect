module {
  func.func @main() -> i32 attributes { exported } {
    %c0 = arith.constant 0 : index
    %c800 = arith.constant 800 : index
    %c1200 = arith.constant 1200 : index
    %cst = arith.constant 1.100000e+03 : f64
    %cst_0 = arith.constant 1.200000e+03 : f64
    %cst_1 = arith.constant 9.000000e+02 : f64
    %cst_2 = arith.constant 8.000000e+02 : f64
    %cst_3 = arith.constant 0.000000e+00 : f64
    %cst_4 = arith.constant 1.500000e+00 : f64
    %cst_5 = arith.constant 1.200000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1200_i32 = arith.constant 1200 : i32
    %c1100_i32 = arith.constant 1100 : i32
    %c900_i32 = arith.constant 900 : i32
    %c800_i32 = arith.constant 800 : i32
    %alloc = memref.alloc() : memref<800x900xf64>
    %alloc_6 = memref.alloc() : memref<800x1200xf64>
    scf.for %arg0 = %c0 to %c800 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c1200 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.addi %1, %c2_i32 : i32
        %3 = arith.muli %0, %2 : i32
        %4 = arith.remsi %3, %c1100_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        memref.store %6, %alloc_6[%arg0, %arg1] : memref<800x1200xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 800 {
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 900 {
        affine.store %cst_3, %alloc[%arg0, %arg1] : memref<800x900xf64>
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.addi %1, %c1_i32 : i32
        affine.for %arg2 = 0 to 1100 {
          %3 = arith.index_cast %arg2 : index to i32
          %4 = arith.muli %0, %3 : i32
          %5 = arith.addi %4, %c1_i32 : i32
          %6 = arith.remsi %5, %c800_i32 : i32
          %7 = arith.sitofp %6 : i32 to f64
          %8 = arith.divf %7, %cst_2 : f64
          %9 = arith.mulf %8, %cst_4 : f64
          %10 = arith.muli %3, %2 : i32
          %11 = arith.remsi %10, %c900_i32 : i32
          %12 = arith.sitofp %11 : i32 to f64
          %13 = arith.divf %12, %cst_1 : f64
          %14 = arith.mulf %9, %13 : f64
          %15 = affine.load %alloc[%arg0, %arg1] : memref<800x900xf64>
          %16 = arith.addf %15, %14 : f64
          affine.store %16, %alloc[%arg0, %arg1] : memref<800x900xf64>
        }
      }
    }
    affine.for %arg0 = 0 to 800 {
      affine.for %arg1 = 0 to 1200 {
        %0 = affine.load %alloc_6[%arg0, %arg1] : memref<800x1200xf64>
        %1 = arith.mulf %0, %cst_5 : f64
        affine.store %1, %alloc_6[%arg0, %arg1] : memref<800x1200xf64>
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.addi %2, %c3_i32 : i32
        affine.for %arg2 = 0 to 900 {
          %4 = affine.load %alloc[%arg0, %arg2] : memref<800x900xf64>
          %5 = arith.index_cast %arg2 : index to i32
          %6 = arith.muli %5, %3 : i32
          %7 = arith.addi %6, %c1_i32 : i32
          %8 = arith.remsi %7, %c1200_i32 : i32
          %9 = arith.sitofp %8 : i32 to f64
          %10 = arith.divf %9, %cst_0 : f64
          %11 = arith.mulf %4, %10 : f64
          %12 = affine.load %alloc_6[%arg0, %arg1] : memref<800x1200xf64>
          %13 = arith.addf %12, %11 : f64
          affine.store %13, %alloc_6[%arg0, %arg1] : memref<800x1200xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c800 step %c1 {
      scf.for %arg1 = %c0 to %c1200 step %c1 {
        %0 = memref.load %alloc_6[%arg0, %arg1] : memref<800x1200xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<800x900xf64>
    memref.dealloc %alloc_6 : memref<800x1200xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
