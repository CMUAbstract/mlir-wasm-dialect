module {
  func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {
    %c40 = arith.constant 40 : index
    %c70 = arith.constant 70 : index
    %cst = arith.constant 3.000000e+02 : f64
    %cst_0 = arith.constant 3.500000e+02 : f64
    %cst_1 = arith.constant 2.500000e+02 : f64
    %cst_2 = arith.constant 2.000000e+02 : f64
    %cst_3 = arith.constant 0.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c70_i32 = arith.constant 70 : i32
    %c60_i32 = arith.constant 60 : i32
    %c50_i32 = arith.constant 50 : i32
    %c40_i32 = arith.constant 40 : i32
    %alloc = memref.alloc() : memref<40x50xf64>
    %alloc_4 = memref.alloc() : memref<50x70xf64>
    %alloc_5 = memref.alloc() : memref<40x70xf64>
    affine.for %arg2 = 0 to 40 {
      %0 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 50 {
        affine.store %cst_3, %alloc[%arg2, %arg3] : memref<40x50xf64>
        %1 = arith.index_cast %arg3 : index to i32
        %2 = arith.addi %1, %c1_i32 : i32
        affine.for %arg4 = 0 to 60 {
          %3 = arith.index_cast %arg4 : index to i32
          %4 = arith.muli %0, %3 : i32
          %5 = arith.addi %4, %c1_i32 : i32
          %6 = arith.remsi %5, %c40_i32 : i32
          %7 = arith.sitofp %6 : i32 to f64
          %8 = arith.divf %7, %cst_2 : f64
          %9 = arith.muli %3, %2 : i32
          %10 = arith.addi %9, %c2_i32 : i32
          %11 = arith.remsi %10, %c50_i32 : i32
          %12 = arith.sitofp %11 : i32 to f64
          %13 = arith.divf %12, %cst_1 : f64
          %14 = arith.mulf %8, %13 : f64
          %15 = affine.load %alloc[%arg2, %arg3] : memref<40x50xf64>
          %16 = arith.addf %15, %14 : f64
          affine.store %16, %alloc[%arg2, %arg3] : memref<40x50xf64>
        }
      }
    }
    affine.for %arg2 = 0 to 50 {
      %0 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 70 {
        affine.store %cst_3, %alloc_4[%arg2, %arg3] : memref<50x70xf64>
        %1 = arith.index_cast %arg3 : index to i32
        %2 = arith.addi %1, %c2_i32 : i32
        affine.for %arg4 = 0 to 80 {
          %3 = arith.index_cast %arg4 : index to i32
          %4 = arith.addi %3, %c3_i32 : i32
          %5 = arith.muli %0, %4 : i32
          %6 = arith.remsi %5, %c70_i32 : i32
          %7 = arith.sitofp %6 : i32 to f64
          %8 = arith.divf %7, %cst_0 : f64
          %9 = arith.muli %3, %2 : i32
          %10 = arith.addi %9, %c2_i32 : i32
          %11 = arith.remsi %10, %c60_i32 : i32
          %12 = arith.sitofp %11 : i32 to f64
          %13 = arith.divf %12, %cst : f64
          %14 = arith.mulf %8, %13 : f64
          %15 = affine.load %alloc_4[%arg2, %arg3] : memref<50x70xf64>
          %16 = arith.addf %15, %14 : f64
          affine.store %16, %alloc_4[%arg2, %arg3] : memref<50x70xf64>
        }
      }
    }
    affine.for %arg2 = 0 to 40 {
      affine.for %arg3 = 0 to 70 {
        affine.store %cst_3, %alloc_5[%arg2, %arg3] : memref<40x70xf64>
        affine.for %arg4 = 0 to 50 {
          %0 = affine.load %alloc[%arg2, %arg4] : memref<40x50xf64>
          %1 = affine.load %alloc_4[%arg4, %arg3] : memref<50x70xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %alloc_5[%arg2, %arg3] : memref<40x70xf64>
          %4 = arith.addf %3, %2 : f64
          affine.store %4, %alloc_5[%arg2, %arg3] : memref<40x70xf64>
        }
      }
    }
    scf.for %arg2 = %c0 to %c40 step %c1 {
      scf.for %arg3 = %c0 to %c70 step %c1 {
        %0 = memref.load %alloc_5[%arg2, %arg3] : memref<40x70xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<40x50xf64>
    memref.dealloc %alloc_4 : memref<50x70xf64>
    memref.dealloc %alloc_5 : memref<40x70xf64>
    return %c0_i32 : i32
  }
  func.func private @print_i32(i32)
}
