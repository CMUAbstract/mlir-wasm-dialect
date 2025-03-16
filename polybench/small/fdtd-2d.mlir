module {
  func.func @main() -> i32 {
    %c60 = arith.constant 60 : index
    %c80 = arith.constant 80 : index
    %cst = arith.constant 8.000000e+01 : f64
    %cst_0 = arith.constant 6.000000e+01 : f64
    %cst_1 = arith.constant 0.69999999999999996 : f64
    %cst_2 = arith.constant 5.000000e-01 : f64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<60x80xf64>
    %alloc_3 = memref.alloc() : memref<60x80xf64>
    %alloc_4 = memref.alloc() : memref<60x80xf64>
    scf.for %arg0 = %c0 to %c60 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.sitofp %0 : i32 to f64
      scf.for %arg1 = %c0 to %c80 step %c1 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.addi %2, %c1_i32 : i32
        %4 = arith.sitofp %3 : i32 to f64
        %5 = arith.mulf %1, %4 : f64
        %6 = arith.divf %5, %cst_0 : f64
        memref.store %6, %alloc[%arg0, %arg1] : memref<60x80xf64>
        %7 = arith.addi %2, %c2_i32 : i32
        %8 = arith.sitofp %7 : i32 to f64
        %9 = arith.mulf %1, %8 : f64
        %10 = arith.divf %9, %cst : f64
        memref.store %10, %alloc_3[%arg0, %arg1] : memref<60x80xf64>
        %11 = arith.addi %2, %c3_i32 : i32
        %12 = arith.sitofp %11 : i32 to f64
        %13 = arith.mulf %1, %12 : f64
        %14 = arith.divf %13, %cst_0 : f64
        memref.store %14, %alloc_4[%arg0, %arg1] : memref<60x80xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 40 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.sitofp %0 : i32 to f64
      affine.for %arg1 = 0 to 80 {
        affine.store %1, %alloc_3[0, %arg1] : memref<60x80xf64>
      }
      affine.for %arg1 = 1 to 60 {
        affine.for %arg2 = 0 to 80 {
          %2 = affine.load %alloc_3[%arg1, %arg2] : memref<60x80xf64>
          %3 = affine.load %alloc_4[%arg1, %arg2] : memref<60x80xf64>
          %4 = affine.load %alloc_4[%arg1 - 1, %arg2] : memref<60x80xf64>
          %5 = arith.subf %3, %4 : f64
          %6 = arith.mulf %5, %cst_2 : f64
          %7 = arith.subf %2, %6 : f64
          affine.store %7, %alloc_3[%arg1, %arg2] : memref<60x80xf64>
        }
      }
      affine.for %arg1 = 0 to 60 {
        affine.for %arg2 = 1 to 80 {
          %2 = affine.load %alloc[%arg1, %arg2] : memref<60x80xf64>
          %3 = affine.load %alloc_4[%arg1, %arg2] : memref<60x80xf64>
          %4 = affine.load %alloc_4[%arg1, %arg2 - 1] : memref<60x80xf64>
          %5 = arith.subf %3, %4 : f64
          %6 = arith.mulf %5, %cst_2 : f64
          %7 = arith.subf %2, %6 : f64
          affine.store %7, %alloc[%arg1, %arg2] : memref<60x80xf64>
        }
      }
      affine.for %arg1 = 0 to 59 {
        affine.for %arg2 = 0 to 79 {
          %2 = affine.load %alloc_4[%arg1, %arg2] : memref<60x80xf64>
          %3 = affine.load %alloc[%arg1, %arg2 + 1] : memref<60x80xf64>
          %4 = affine.load %alloc[%arg1, %arg2] : memref<60x80xf64>
          %5 = arith.subf %3, %4 : f64
          %6 = affine.load %alloc_3[%arg1 + 1, %arg2] : memref<60x80xf64>
          %7 = arith.addf %5, %6 : f64
          %8 = affine.load %alloc_3[%arg1, %arg2] : memref<60x80xf64>
          %9 = arith.subf %7, %8 : f64
          %10 = arith.mulf %9, %cst_1 : f64
          %11 = arith.subf %2, %10 : f64
          affine.store %11, %alloc_4[%arg1, %arg2] : memref<60x80xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c60 step %c1 {
      scf.for %arg1 = %c0 to %c80 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<60x80xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    scf.for %arg0 = %c0 to %c60 step %c1 {
      scf.for %arg1 = %c0 to %c80 step %c1 {
        %0 = memref.load %alloc_3[%arg0, %arg1] : memref<60x80xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    scf.for %arg0 = %c0 to %c60 step %c1 {
      scf.for %arg1 = %c0 to %c80 step %c1 {
        %0 = memref.load %alloc_4[%arg0, %arg1] : memref<60x80xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<60x80xf64>
    memref.dealloc %alloc_3 : memref<60x80xf64>
    memref.dealloc %alloc_4 : memref<60x80xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
