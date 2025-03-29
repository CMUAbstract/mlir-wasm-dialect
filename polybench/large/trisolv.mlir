#map = affine_map<(d0) -> (d0)>
module {
  func.func @main() -> i32 {
    %cst = arith.constant 2.001000e+00 : f64
    %c2000 = arith.constant 2000 : index
    %cst_0 = arith.constant 2.000000e+03 : f64
    %cst_1 = arith.constant -9.990000e+02 : f64
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant 2.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2000_i32 = arith.constant 2000 : i32
    %alloc = memref.alloc() : memref<2000xf64>
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      memref.store %cst_1, %alloc[%arg0] : memref<2000xf64>
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 2000 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.sitofp %0 : i32 to f64
      affine.store %1, %alloc[%arg0] : memref<2000xf64>
      %2 = arith.addi %0, %c2000_i32 : i32
      affine.for %arg1 = 0 to #map(%arg0) {
        %5 = arith.index_cast %arg1 : index to i32
        %6 = arith.subi %2, %5 : i32
        %7 = arith.addi %6, %c1_i32 : i32
        %8 = arith.sitofp %7 : i32 to f64
        %9 = arith.mulf %8, %cst_2 : f64
        %10 = arith.divf %9, %cst_0 : f64
        %11 = affine.load %alloc[%arg1] : memref<2000xf64>
        %12 = arith.mulf %10, %11 : f64
        %13 = affine.load %alloc[%arg0] : memref<2000xf64>
        %14 = arith.subf %13, %12 : f64
        affine.store %14, %alloc[%arg0] : memref<2000xf64>
      }
      %3 = affine.load %alloc[%arg0] : memref<2000xf64>
      %4 = arith.divf %3, %cst : f64
      affine.store %4, %alloc[%arg0] : memref<2000xf64>
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      %0 = memref.load %alloc[%arg0] : memref<2000xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<2000xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
