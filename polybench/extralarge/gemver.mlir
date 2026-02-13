module {
  func.func @main() -> i32 attributes { exported } {
    %c0 = arith.constant 0 : index
    %c4000 = arith.constant 4000 : index
    %cst = arith.constant 4.000000e+03 : f64
    %cst_0 = arith.constant 1.500000e+00 : f64
    %cst_1 = arith.constant 1.200000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %cst_2 = arith.constant 2.000000e+00 : f64
    %cst_3 = arith.constant 4.000000e+00 : f64
    %cst_4 = arith.constant 6.000000e+00 : f64
    %cst_5 = arith.constant 8.000000e+00 : f64
    %cst_6 = arith.constant 9.000000e+00 : f64
    %cst_7 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c4000_i32 = arith.constant 4000 : i32
    %alloc = memref.alloc() : memref<4000x4000xf64>
    %alloc_8 = memref.alloc() : memref<4000xf64>
    %alloc_9 = memref.alloc() : memref<4000xf64>
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      memref.store %cst_7, %alloc_9[%arg0] : memref<4000xf64>
      memref.store %cst_7, %alloc_8[%arg0] : memref<4000xf64>
      scf.for %arg1 = %c0 to %c4000 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.muli %0, %1 : i32
        %3 = arith.remsi %2, %c4000_i32 : i32
        %4 = arith.sitofp %3 : i32 to f64
        %5 = arith.divf %4, %cst : f64
        memref.store %5, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 4000 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.sitofp %0 : i32 to f64
      %2 = arith.addi %0, %c1_i32 : i32
      %3 = arith.sitofp %2 : i32 to f64
      %4 = arith.divf %3, %cst : f64
      %5 = arith.divf %4, %cst_2 : f64
      affine.for %arg1 = 0 to 4000 {
        %6 = affine.load %alloc[%arg0, %arg1] : memref<4000x4000xf64>
        %7 = arith.index_cast %arg1 : index to i32
        %8 = arith.addi %7, %c1_i32 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst : f64
        %11 = arith.divf %10, %cst_3 : f64
        %12 = arith.mulf %1, %11 : f64
        %13 = arith.addf %6, %12 : f64
        %14 = arith.divf %10, %cst_4 : f64
        %15 = arith.mulf %5, %14 : f64
        %16 = arith.addf %13, %15 : f64
        affine.store %16, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
    }
    affine.for %arg0 = 0 to 4000 {
      affine.for %arg1 = 0 to 4000 {
        %0 = affine.load %alloc_9[%arg0] : memref<4000xf64>
        %1 = affine.load %alloc[%arg1, %arg0] : memref<4000x4000xf64>
        %2 = arith.mulf %1, %cst_1 : f64
        %3 = arith.index_cast %arg1 : index to i32
        %4 = arith.addi %3, %c1_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        %7 = arith.divf %6, %cst_5 : f64
        %8 = arith.mulf %2, %7 : f64
        %9 = arith.addf %0, %8 : f64
        affine.store %9, %alloc_9[%arg0] : memref<4000xf64>
      }
    }
    affine.for %arg0 = 0 to 4000 {
      %0 = affine.load %alloc_9[%arg0] : memref<4000xf64>
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.addi %1, %c1_i32 : i32
      %3 = arith.sitofp %2 : i32 to f64
      %4 = arith.divf %3, %cst : f64
      %5 = arith.divf %4, %cst_6 : f64
      %6 = arith.addf %0, %5 : f64
      affine.store %6, %alloc_9[%arg0] : memref<4000xf64>
    }
    affine.for %arg0 = 0 to 4000 {
      affine.for %arg1 = 0 to 4000 {
        %0 = affine.load %alloc_8[%arg0] : memref<4000xf64>
        %1 = affine.load %alloc[%arg0, %arg1] : memref<4000x4000xf64>
        %2 = arith.mulf %1, %cst_0 : f64
        %3 = affine.load %alloc_9[%arg1] : memref<4000xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = arith.addf %0, %4 : f64
        affine.store %5, %alloc_8[%arg0] : memref<4000xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %0 = memref.load %alloc_8[%arg0] : memref<4000xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<4000x4000xf64>
    memref.dealloc %alloc_8 : memref<4000xf64>
    memref.dealloc %alloc_9 : memref<4000xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
