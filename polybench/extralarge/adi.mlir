module {
  func.func @main() -> i32 {
    %c2000 = arith.constant 2000 : index
    %cst = arith.constant -7999.0000000000009 : f64
    %cst_0 = arith.constant -3999.0000000000005 : f64
    %cst_1 = arith.constant -4000.0000000000005 : f64
    %cst_2 = arith.constant 2000.0000000000002 : f64
    %cst_3 = arith.constant 4000.0000000000005 : f64
    %cst_4 = arith.constant 4001.0000000000005 : f64
    %cst_5 = arith.constant -2000.0000000000002 : f64
    %cst_6 = arith.constant 8001.0000000000009 : f64
    %cst_7 = arith.constant 2.000000e+03 : f64
    %cst_8 = arith.constant 0.000000e+00 : f64
    %cst_9 = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2000_i32 = arith.constant 2000 : i32
    %alloc = memref.alloc() : memref<2000x2000xf64>
    %alloc_10 = memref.alloc() : memref<2000x2000xf64>
    %alloc_11 = memref.alloc() : memref<2000x2000xf64>
    %alloc_12 = memref.alloc() : memref<2000x2000xf64>
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.addi %0, %c2000_i32 : i32
      scf.for %arg1 = %c0 to %c2000 step %c1 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.subi %1, %2 : i32
        %4 = arith.sitofp %3 : i32 to f64
        %5 = arith.divf %4, %cst_7 : f64
        memref.store %5, %alloc[%arg0, %arg1] : memref<2000x2000xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 1 to 1001 {
      affine.for %arg1 = 1 to 1999 {
        affine.store %cst_9, %alloc_10[0, %arg1] : memref<2000x2000xf64>
        affine.store %cst_8, %alloc_11[%arg1, 0] : memref<2000x2000xf64>
        affine.store %cst_9, %alloc_12[%arg1, 0] : memref<2000x2000xf64>
        affine.for %arg2 = 1 to 1999 {
          %0 = affine.load %alloc_11[%arg1, %arg2 - 1] : memref<2000x2000xf64>
          %1 = arith.mulf %0, %cst_1 : f64
          %2 = arith.addf %1, %cst_6 : f64
          %3 = arith.divf %cst_3, %2 : f64
          affine.store %3, %alloc_11[%arg1, %arg2] : memref<2000x2000xf64>
          %4 = affine.load %alloc[%arg2, %arg1 - 1] : memref<2000x2000xf64>
          %5 = arith.mulf %4, %cst_2 : f64
          %6 = affine.load %alloc[%arg2, %arg1] : memref<2000x2000xf64>
          %7 = arith.mulf %6, %cst_0 : f64
          %8 = arith.addf %5, %7 : f64
          %9 = affine.load %alloc[%arg2, %arg1 + 1] : memref<2000x2000xf64>
          %10 = arith.mulf %9, %cst_5 : f64
          %11 = arith.subf %8, %10 : f64
          %12 = affine.load %alloc_12[%arg1, %arg2 - 1] : memref<2000x2000xf64>
          %13 = arith.mulf %12, %cst_1 : f64
          %14 = arith.subf %11, %13 : f64
          %15 = arith.divf %14, %2 : f64
          affine.store %15, %alloc_12[%arg1, %arg2] : memref<2000x2000xf64>
        }
        affine.store %cst_9, %alloc_10[1999, %arg1] : memref<2000x2000xf64>
        affine.for %arg2 = 1 to 1999 {
          %0 = affine.load %alloc_11[%arg1, -%arg2 + 1999] : memref<2000x2000xf64>
          %1 = affine.load %alloc_10[-%arg2 + 2000, %arg1] : memref<2000x2000xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %alloc_12[%arg1, -%arg2 + 1999] : memref<2000x2000xf64>
          %4 = arith.addf %2, %3 : f64
          affine.store %4, %alloc_10[-%arg2 + 1999, %arg1] : memref<2000x2000xf64>
        }
      }
      affine.for %arg1 = 1 to 1999 {
        affine.store %cst_9, %alloc[%arg1, 0] : memref<2000x2000xf64>
        affine.store %cst_8, %alloc_11[%arg1, 0] : memref<2000x2000xf64>
        affine.store %cst_9, %alloc_12[%arg1, 0] : memref<2000x2000xf64>
        affine.for %arg2 = 1 to 1999 {
          %0 = affine.load %alloc_11[%arg1, %arg2 - 1] : memref<2000x2000xf64>
          %1 = arith.mulf %0, %cst_5 : f64
          %2 = arith.addf %1, %cst_4 : f64
          %3 = arith.divf %cst_2, %2 : f64
          affine.store %3, %alloc_11[%arg1, %arg2] : memref<2000x2000xf64>
          %4 = affine.load %alloc_10[%arg1 - 1, %arg2] : memref<2000x2000xf64>
          %5 = arith.mulf %4, %cst_3 : f64
          %6 = affine.load %alloc_10[%arg1, %arg2] : memref<2000x2000xf64>
          %7 = arith.mulf %6, %cst : f64
          %8 = arith.addf %5, %7 : f64
          %9 = affine.load %alloc_10[%arg1 + 1, %arg2] : memref<2000x2000xf64>
          %10 = arith.mulf %9, %cst_1 : f64
          %11 = arith.subf %8, %10 : f64
          %12 = affine.load %alloc_12[%arg1, %arg2 - 1] : memref<2000x2000xf64>
          %13 = arith.mulf %12, %cst_5 : f64
          %14 = arith.subf %11, %13 : f64
          %15 = arith.divf %14, %2 : f64
          affine.store %15, %alloc_12[%arg1, %arg2] : memref<2000x2000xf64>
        }
        affine.store %cst_9, %alloc[%arg1, 1999] : memref<2000x2000xf64>
        affine.for %arg2 = 1 to 1999 {
          %0 = affine.load %alloc_11[%arg1, -%arg2 + 1999] : memref<2000x2000xf64>
          %1 = affine.load %alloc[%arg1, -%arg2 + 2000] : memref<2000x2000xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %alloc_12[%arg1, -%arg2 + 1999] : memref<2000x2000xf64>
          %4 = arith.addf %2, %3 : f64
          affine.store %4, %alloc[%arg1, -%arg2 + 1999] : memref<2000x2000xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      scf.for %arg1 = %c0 to %c2000 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<2000x2000xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<2000x2000xf64>
    memref.dealloc %alloc_10 : memref<2000x2000xf64>
    memref.dealloc %alloc_11 : memref<2000x2000xf64>
    memref.dealloc %alloc_12 : memref<2000x2000xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
