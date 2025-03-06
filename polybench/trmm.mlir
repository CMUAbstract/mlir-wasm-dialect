#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c60 = arith.constant 60 : index
    %c80 = arith.constant 80 : index
    %cst = arith.constant 8.000000e+01 : f64
    %cst_0 = arith.constant 6.000000e+01 : f64
    %cst_1 = arith.constant 1.500000e+00 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c80_i32 = arith.constant 80 : i32
    %c60_i32 = arith.constant 60 : i32
    %alloc = memref.alloc() : memref<60x60xf64>
    %alloc_3 = memref.alloc() : memref<60x80xf64>
    scf.for %arg0 = %c0 to %c60 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %arg0 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.addi %0, %1 : i32
        %3 = arith.remsi %2, %c60_i32 : i32
        %4 = arith.sitofp %3 : i32 to f64
        %5 = arith.divf %4, %cst_0 : f64
        memref.store %5, %alloc[%arg0, %arg1] : memref<60x60xf64>
      }
      memref.store %cst_2, %alloc[%arg0, %arg0] : memref<60x60xf64>
      scf.for %arg1 = %c0 to %c80 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.subi %0, %1 : i32
        %3 = arith.addi %2, %c80_i32 : i32
        %4 = arith.remsi %3, %c80_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        memref.store %6, %alloc_3[%arg0, %arg1] : memref<60x80xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 60 {
      affine.for %arg1 = 0 to 80 {
        affine.for %arg2 = #map(%arg0) to 60 {
          %2 = affine.load %alloc[%arg2, %arg0] : memref<60x60xf64>
          %3 = affine.load %alloc_3[%arg2, %arg1] : memref<60x80xf64>
          %4 = arith.mulf %2, %3 : f64
          %5 = affine.load %alloc_3[%arg0, %arg1] : memref<60x80xf64>
          %6 = arith.addf %5, %4 : f64
          affine.store %6, %alloc_3[%arg0, %arg1] : memref<60x80xf64>
        }
        %0 = affine.load %alloc_3[%arg0, %arg1] : memref<60x80xf64>
        %1 = arith.mulf %0, %cst_1 : f64
        affine.store %1, %alloc_3[%arg0, %arg1] : memref<60x80xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c60 step %c1 {
      scf.for %arg1 = %c0 to %c80 step %c1 {
        %0 = memref.load %alloc_3[%arg0, %arg1] : memref<60x80xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<60x60xf64>
    memref.dealloc %alloc_3 : memref<60x80xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
