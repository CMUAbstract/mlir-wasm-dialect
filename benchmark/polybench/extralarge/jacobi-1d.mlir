module {
  func.func @main() -> i32 attributes { exported } {
    %c4000 = arith.constant 4000 : index
    %cst = arith.constant 4.000000e+03 : f64
    %cst_0 = arith.constant 3.333300e-01 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<4000xf64>
    %alloc_3 = memref.alloc() : memref<4000xf64>
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.sitofp %0 : i32 to f64
      %2 = arith.addf %1, %cst_1 : f64
      %3 = arith.divf %2, %cst : f64
      memref.store %3, %alloc[%arg0] : memref<4000xf64>
      %4 = arith.addf %1, %cst_2 : f64
      %5 = arith.divf %4, %cst : f64
      memref.store %5, %alloc_3[%arg0] : memref<4000xf64>
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 1000 {
      affine.for %arg1 = 1 to 3999 {
        %0 = affine.load %alloc[%arg1 - 1] : memref<4000xf64>
        %1 = affine.load %alloc[%arg1] : memref<4000xf64>
        %2 = arith.addf %0, %1 : f64
        %3 = affine.load %alloc[%arg1 + 1] : memref<4000xf64>
        %4 = arith.addf %2, %3 : f64
        %5 = arith.mulf %4, %cst_0 : f64
        affine.store %5, %alloc_3[%arg1] : memref<4000xf64>
      }
      affine.for %arg1 = 1 to 3999 {
        %0 = affine.load %alloc_3[%arg1 - 1] : memref<4000xf64>
        %1 = affine.load %alloc_3[%arg1] : memref<4000xf64>
        %2 = arith.addf %0, %1 : f64
        %3 = affine.load %alloc_3[%arg1 + 1] : memref<4000xf64>
        %4 = arith.addf %2, %3 : f64
        %5 = arith.mulf %4, %cst_0 : f64
        affine.store %5, %alloc[%arg1] : memref<4000xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %0 = memref.load %alloc[%arg0] : memref<4000xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<4000xf64>
    memref.dealloc %alloc_3 : memref<4000xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
