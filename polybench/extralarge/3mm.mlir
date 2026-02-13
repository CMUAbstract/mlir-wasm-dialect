module {
  func.func @main() -> i32 attributes { exported } {
    %c1600 = arith.constant 1600 : index
    %c2200 = arith.constant 2200 : index
    %cst = arith.constant 1.000000e+04 : f64
    %cst_0 = arith.constant 1.100000e+04 : f64
    %cst_1 = arith.constant 9.000000e+03 : f64
    %cst_2 = arith.constant 8.000000e+03 : f64
    %cst_3 = arith.constant 0.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2200_i32 = arith.constant 2200 : i32
    %c2000_i32 = arith.constant 2000 : i32
    %c1800_i32 = arith.constant 1800 : i32
    %c1600_i32 = arith.constant 1600 : i32
    %alloc = memref.alloc() : memref<1600x1800xf64>
    %alloc_4 = memref.alloc() : memref<1800x2200xf64>
    %alloc_5 = memref.alloc() : memref<1600x2200xf64>
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 1600 {
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 1800 {
        affine.store %cst_3, %alloc[%arg0, %arg1] : memref<1600x1800xf64>
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.addi %1, %c1_i32 : i32
        affine.for %arg2 = 0 to 2000 {
          %3 = arith.index_cast %arg2 : index to i32
          %4 = arith.muli %0, %3 : i32
          %5 = arith.addi %4, %c1_i32 : i32
          %6 = arith.remsi %5, %c1600_i32 : i32
          %7 = arith.sitofp %6 : i32 to f64
          %8 = arith.divf %7, %cst_2 : f64
          %9 = arith.muli %3, %2 : i32
          %10 = arith.addi %9, %c2_i32 : i32
          %11 = arith.remsi %10, %c1800_i32 : i32
          %12 = arith.sitofp %11 : i32 to f64
          %13 = arith.divf %12, %cst_1 : f64
          %14 = arith.mulf %8, %13 : f64
          %15 = affine.load %alloc[%arg0, %arg1] : memref<1600x1800xf64>
          %16 = arith.addf %15, %14 : f64
          affine.store %16, %alloc[%arg0, %arg1] : memref<1600x1800xf64>
        }
      }
    }
    affine.for %arg0 = 0 to 1800 {
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 2200 {
        affine.store %cst_3, %alloc_4[%arg0, %arg1] : memref<1800x2200xf64>
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.addi %1, %c2_i32 : i32
        affine.for %arg2 = 0 to 2400 {
          %3 = arith.index_cast %arg2 : index to i32
          %4 = arith.addi %3, %c3_i32 : i32
          %5 = arith.muli %0, %4 : i32
          %6 = arith.remsi %5, %c2200_i32 : i32
          %7 = arith.sitofp %6 : i32 to f64
          %8 = arith.divf %7, %cst_0 : f64
          %9 = arith.muli %3, %2 : i32
          %10 = arith.addi %9, %c2_i32 : i32
          %11 = arith.remsi %10, %c2000_i32 : i32
          %12 = arith.sitofp %11 : i32 to f64
          %13 = arith.divf %12, %cst : f64
          %14 = arith.mulf %8, %13 : f64
          %15 = affine.load %alloc_4[%arg0, %arg1] : memref<1800x2200xf64>
          %16 = arith.addf %15, %14 : f64
          affine.store %16, %alloc_4[%arg0, %arg1] : memref<1800x2200xf64>
        }
      }
    }
    affine.for %arg0 = 0 to 1600 {
      affine.for %arg1 = 0 to 2200 {
        affine.store %cst_3, %alloc_5[%arg0, %arg1] : memref<1600x2200xf64>
        affine.for %arg2 = 0 to 1800 {
          %0 = affine.load %alloc[%arg0, %arg2] : memref<1600x1800xf64>
          %1 = affine.load %alloc_4[%arg2, %arg1] : memref<1800x2200xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %alloc_5[%arg0, %arg1] : memref<1600x2200xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %alloc_5[%arg0, %arg1] : memref<1600x2200xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c1600 step %c1 {
      scf.for %arg1 = %c0 to %c2200 step %c1 {
        %0 = memref.load %alloc_5[%arg0, %arg1] : memref<1600x2200xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<1600x1800xf64>
    memref.dealloc %alloc_4 : memref<1800x2200xf64>
    memref.dealloc %alloc_5 : memref<1600x2200xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
