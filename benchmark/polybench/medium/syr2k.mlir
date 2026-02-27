#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @main() -> i32 attributes { exported } {
    %c0 = arith.constant 0 : index
    %c240 = arith.constant 240 : index
    %cst = arith.constant 2.000000e+02 : f64
    %cst_0 = arith.constant 2.400000e+02 : f64
    %cst_1 = arith.constant 1.500000e+00 : f64
    %cst_2 = arith.constant 1.200000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c200_i32 = arith.constant 200 : i32
    %c240_i32 = arith.constant 240 : i32
    %alloc = memref.alloc() : memref<240x240xf64>
    scf.for %arg0 = %c0 to %c240 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c240 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.muli %0, %1 : i32
        %3 = arith.addi %2, %c3_i32 : i32
        %4 = arith.remsi %3, %c240_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        memref.store %6, %alloc[%arg0, %arg1] : memref<240x240xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 240 {
      affine.for %arg1 = 0 to #map(%arg0) {
        %1 = affine.load %alloc[%arg0, %arg1] : memref<240x240xf64>
        %2 = arith.mulf %1, %cst_2 : f64
        affine.store %2, %alloc[%arg0, %arg1] : memref<240x240xf64>
      }
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 200 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.muli %0, %1 : i32
        %3 = arith.addi %2, %c2_i32 : i32
        %4 = arith.remsi %3, %c200_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        %7 = arith.addi %2, %c1_i32 : i32
        %8 = arith.remsi %7, %c240_i32 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst_0 : f64
        affine.for %arg2 = 0 to #map(%arg0) {
          %11 = arith.index_cast %arg2 : index to i32
          %12 = arith.muli %11, %1 : i32
          %13 = arith.addi %12, %c1_i32 : i32
          %14 = arith.remsi %13, %c240_i32 : i32
          %15 = arith.sitofp %14 : i32 to f64
          %16 = arith.divf %15, %cst_0 : f64
          %17 = arith.mulf %16, %cst_1 : f64
          %18 = arith.mulf %17, %6 : f64
          %19 = arith.addi %12, %c2_i32 : i32
          %20 = arith.remsi %19, %c200_i32 : i32
          %21 = arith.sitofp %20 : i32 to f64
          %22 = arith.divf %21, %cst : f64
          %23 = arith.mulf %22, %cst_1 : f64
          %24 = arith.mulf %23, %10 : f64
          %25 = arith.addf %18, %24 : f64
          %26 = affine.load %alloc[%arg0, %arg2] : memref<240x240xf64>
          %27 = arith.addf %26, %25 : f64
          affine.store %27, %alloc[%arg0, %arg2] : memref<240x240xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c240 step %c1 {
      scf.for %arg1 = %c0 to %c240 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<240x240xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<240x240xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
