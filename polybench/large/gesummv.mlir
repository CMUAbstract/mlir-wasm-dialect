module {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c1300 = arith.constant 1300 : index
    %cst = arith.constant 1.300000e+03 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.500000e+00 : f64
    %cst_2 = arith.constant 1.200000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1300_i32 = arith.constant 1300 : i32
    %alloc = memref.alloc() : memref<1300xf64>
    %alloc_3 = memref.alloc() : memref<1300xf64>
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 1300 {
      affine.store %cst_0, %alloc[%arg0] : memref<1300xf64>
      affine.store %cst_0, %alloc_3[%arg0] : memref<1300xf64>
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 1300 {
        %6 = arith.index_cast %arg1 : index to i32
        %7 = arith.muli %0, %6 : i32
        %8 = arith.addi %7, %c1_i32 : i32
        %9 = arith.remsi %8, %c1300_i32 : i32
        %10 = arith.sitofp %9 : i32 to f64
        %11 = arith.divf %10, %cst : f64
        %12 = arith.remsi %6, %c1300_i32 : i32
        %13 = arith.sitofp %12 : i32 to f64
        %14 = arith.divf %13, %cst : f64
        %15 = arith.mulf %11, %14 : f64
        %16 = affine.load %alloc[%arg0] : memref<1300xf64>
        %17 = arith.addf %15, %16 : f64
        affine.store %17, %alloc[%arg0] : memref<1300xf64>
        %18 = arith.addi %7, %c2_i32 : i32
        %19 = arith.remsi %18, %c1300_i32 : i32
        %20 = arith.sitofp %19 : i32 to f64
        %21 = arith.divf %20, %cst : f64
        %22 = arith.mulf %21, %14 : f64
        %23 = affine.load %alloc_3[%arg0] : memref<1300xf64>
        %24 = arith.addf %22, %23 : f64
        affine.store %24, %alloc_3[%arg0] : memref<1300xf64>
      }
      %1 = affine.load %alloc[%arg0] : memref<1300xf64>
      %2 = arith.mulf %1, %cst_1 : f64
      %3 = affine.load %alloc_3[%arg0] : memref<1300xf64>
      %4 = arith.mulf %3, %cst_2 : f64
      %5 = arith.addf %2, %4 : f64
      affine.store %5, %alloc_3[%arg0] : memref<1300xf64>
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c1300 step %c1 {
      %0 = memref.load %alloc_3[%arg0] : memref<1300xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<1300xf64>
    memref.dealloc %alloc_3 : memref<1300xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
