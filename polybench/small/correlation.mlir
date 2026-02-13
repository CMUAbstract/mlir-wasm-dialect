#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @main() -> i32 attributes { exported } {
    %cst = arith.constant 1.000000e+01 : f64
    %c0 = arith.constant 0 : index
    %c80 = arith.constant 80 : index
    %c100 = arith.constant 100 : index
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e-01 : f64
    %cst_3 = arith.constant 1.000000e+02 : f64
    %cst_4 = arith.constant 8.000000e+01 : f64
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<100x80xf64>
    %alloc_5 = memref.alloc() : memref<80x80xf64>
    %alloc_6 = memref.alloc() : memref<80xf64>
    %alloc_7 = memref.alloc() : memref<80xf64>
    scf.for %arg0 = %c0 to %c100 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.sitofp %0 : i32 to f64
      scf.for %arg1 = %c0 to %c80 step %c1 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.muli %0, %2 : i32
        %4 = arith.sitofp %3 : i32 to f64
        %5 = arith.divf %4, %cst_4 : f64
        %6 = arith.addf %5, %1 : f64
        memref.store %6, %alloc[%arg0, %arg1] : memref<100x80xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 80 {
      affine.store %cst_1, %alloc_6[%arg0] : memref<80xf64>
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 100 {
        %3 = arith.index_cast %arg1 : index to i32
        %4 = arith.muli %3, %0 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst_4 : f64
        %7 = arith.sitofp %3 : i32 to f64
        %8 = arith.addf %6, %7 : f64
        %9 = affine.load %alloc_6[%arg0] : memref<80xf64>
        %10 = arith.addf %9, %8 : f64
        affine.store %10, %alloc_6[%arg0] : memref<80xf64>
      }
      %1 = affine.load %alloc_6[%arg0] : memref<80xf64>
      %2 = arith.divf %1, %cst_3 : f64
      affine.store %2, %alloc_6[%arg0] : memref<80xf64>
    }
    affine.for %arg0 = 0 to 80 {
      affine.store %cst_1, %alloc_7[%arg0] : memref<80xf64>
      %0 = arith.index_cast %arg0 : index to i32
      %1 = affine.load %alloc_6[%arg0] : memref<80xf64>
      affine.for %arg1 = 0 to 100 {
        %7 = arith.index_cast %arg1 : index to i32
        %8 = arith.muli %7, %0 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst_4 : f64
        %11 = arith.sitofp %7 : i32 to f64
        %12 = arith.addf %10, %11 : f64
        %13 = arith.subf %12, %1 : f64
        %14 = arith.mulf %13, %13 : f64
        %15 = affine.load %alloc_7[%arg0] : memref<80xf64>
        %16 = arith.addf %15, %14 : f64
        affine.store %16, %alloc_7[%arg0] : memref<80xf64>
      }
      %2 = affine.load %alloc_7[%arg0] : memref<80xf64>
      %3 = arith.divf %2, %cst_3 : f64
      %4 = math.sqrt %3 : f64
      %5 = arith.cmpf ole, %4, %cst_2 : f64
      %6 = arith.select %5, %cst_0, %4 : f64
      affine.store %6, %alloc_7[%arg0] : memref<80xf64>
    }
    affine.for %arg0 = 0 to 100 {
      affine.for %arg1 = 0 to 80 {
        %0 = affine.load %alloc_6[%arg1] : memref<80xf64>
        %1 = affine.load %alloc[%arg0, %arg1] : memref<100x80xf64>
        %2 = arith.subf %1, %0 : f64
        affine.store %2, %alloc[%arg0, %arg1] : memref<100x80xf64>
        %3 = affine.load %alloc_7[%arg1] : memref<80xf64>
        %4 = arith.mulf %3, %cst : f64
        %5 = arith.divf %2, %4 : f64
        affine.store %5, %alloc[%arg0, %arg1] : memref<100x80xf64>
      }
    }
    affine.for %arg0 = 0 to 79 {
      affine.store %cst_0, %alloc_5[%arg0, %arg0] : memref<80x80xf64>
      affine.for %arg1 = #map(%arg0) to 80 {
        affine.store %cst_1, %alloc_5[%arg0, %arg1] : memref<80x80xf64>
        affine.for %arg2 = 0 to 100 {
          %1 = affine.load %alloc[%arg2, %arg0] : memref<100x80xf64>
          %2 = affine.load %alloc[%arg2, %arg1] : memref<100x80xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %alloc_5[%arg0, %arg1] : memref<80x80xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %alloc_5[%arg0, %arg1] : memref<80x80xf64>
        }
        %0 = affine.load %alloc_5[%arg0, %arg1] : memref<80x80xf64>
        affine.store %0, %alloc_5[%arg1, %arg0] : memref<80x80xf64>
      }
    }
    affine.store %cst_0, %alloc_5[79, 79] : memref<80x80xf64>
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c80 step %c1 {
      scf.for %arg1 = %c0 to %c80 step %c1 {
        %0 = memref.load %alloc_5[%arg0, %arg1] : memref<80x80xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<100x80xf64>
    memref.dealloc %alloc_5 : memref<80x80xf64>
    memref.dealloc %alloc_6 : memref<80xf64>
    memref.dealloc %alloc_7 : memref<80xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
