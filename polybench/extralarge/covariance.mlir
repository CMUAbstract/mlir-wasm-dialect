#map = affine_map<(d0) -> (d0)>
module {
  func.func @main() -> i32 {
    %cst = arith.constant 2.999000e+03 : f64
    %c0 = arith.constant 0 : index
    %c2600 = arith.constant 2600 : index
    %c3000 = arith.constant 3000 : index
    %cst_0 = arith.constant 3.000000e+03 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 2.600000e+03 : f64
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<3000x2600xf64>
    %alloc_3 = memref.alloc() : memref<2600x2600xf64>
    %alloc_4 = memref.alloc() : memref<2600xf64>
    scf.for %arg0 = %c0 to %c3000 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.sitofp %0 : i32 to f64
      scf.for %arg1 = %c0 to %c2600 step %c1 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.sitofp %2 : i32 to f64
        %4 = arith.mulf %1, %3 : f64
        %5 = arith.divf %4, %cst_2 : f64
        memref.store %5, %alloc[%arg0, %arg1] : memref<3000x2600xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 2600 {
      affine.store %cst_1, %alloc_4[%arg0] : memref<2600xf64>
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.sitofp %0 : i32 to f64
      affine.for %arg1 = 0 to 3000 {
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.mulf %5, %1 : f64
        %7 = arith.divf %6, %cst_2 : f64
        %8 = affine.load %alloc_4[%arg0] : memref<2600xf64>
        %9 = arith.addf %8, %7 : f64
        affine.store %9, %alloc_4[%arg0] : memref<2600xf64>
      }
      %2 = affine.load %alloc_4[%arg0] : memref<2600xf64>
      %3 = arith.divf %2, %cst_0 : f64
      affine.store %3, %alloc_4[%arg0] : memref<2600xf64>
    }
    affine.for %arg0 = 0 to 3000 {
      affine.for %arg1 = 0 to 2600 {
        %0 = affine.load %alloc_4[%arg1] : memref<2600xf64>
        %1 = affine.load %alloc[%arg0, %arg1] : memref<3000x2600xf64>
        %2 = arith.subf %1, %0 : f64
        affine.store %2, %alloc[%arg0, %arg1] : memref<3000x2600xf64>
      }
    }
    affine.for %arg0 = 0 to 2600 {
      affine.for %arg1 = #map(%arg0) to 2600 {
        affine.store %cst_1, %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
        affine.for %arg2 = 0 to 3000 {
          %2 = affine.load %alloc[%arg2, %arg0] : memref<3000x2600xf64>
          %3 = affine.load %alloc[%arg2, %arg1] : memref<3000x2600xf64>
          %4 = arith.mulf %2, %3 : f64
          %5 = affine.load %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
          %6 = arith.addf %5, %4 : f64
          affine.store %6, %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
        }
        %0 = affine.load %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
        %1 = arith.divf %0, %cst : f64
        affine.store %1, %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
        affine.store %1, %alloc_3[%arg1, %arg0] : memref<2600x2600xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c2600 step %c1 {
      scf.for %arg1 = %c0 to %c2600 step %c1 {
        %0 = memref.load %alloc_3[%arg0, %arg1] : memref<2600x2600xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<3000x2600xf64>
    memref.dealloc %alloc_3 : memref<2600x2600xf64>
    memref.dealloc %alloc_4 : memref<2600xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
