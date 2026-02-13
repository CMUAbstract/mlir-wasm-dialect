#map = affine_map<(d0) -> (d0)>
module {
  func.func @main() -> i32 attributes { exported } {
    %c0 = arith.constant 0 : index
    %c200 = arith.constant 200 : index
    %c240 = arith.constant 240 : index
    %cst = arith.constant 2.000000e+02 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.500000e+00 : f64
    %cst_2 = arith.constant 1.200000e+00 : f64
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant -9.990000e+02 : f64
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c240_i32 = arith.constant 240 : i32
    %alloc = memref.alloc() : memref<200x240xf64>
    %alloc_4 = memref.alloc() : memref<200x200xf64>
    scf.for %arg0 = %c0 to %c200 step %c1 {
      %1 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c240 step %c1 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.addi %1, %2 : i32
        %4 = arith.remsi %3, %c100_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        memref.store %6, %alloc[%arg0, %arg1] : memref<200x240xf64>
      }
    }
    scf.for %arg0 = %c0 to %c200 step %c1 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.addi %1, %c1_i32 : i32
      %3 = arith.index_cast %2 : i32 to index
      scf.for %arg1 = %c0 to %3 step %c1 {
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.addi %1, %4 : i32
        %6 = arith.remsi %5, %c100_i32 : i32
        %7 = arith.sitofp %6 : i32 to f64
        %8 = arith.divf %7, %cst : f64
        memref.store %8, %alloc_4[%arg0, %arg1] : memref<200x200xf64>
      }
      scf.for %arg1 = %3 to %c200 step %c1 {
        memref.store %cst_3, %alloc_4[%arg0, %arg1] : memref<200x200xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    %alloca = memref.alloca() : memref<f64>
    affine.for %arg0 = 0 to 200 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.addi %1, %c240_i32 : i32
      %3 = arith.index_cast %arg0 : index to i32
      %4 = arith.addi %3, %c240_i32 : i32
      %5 = affine.load %alloc_4[%arg0, %arg0] : memref<200x200xf64>
      affine.for %arg1 = 0 to 240 {
        affine.store %cst_0, %alloca[] : memref<f64>
        %6 = arith.index_cast %arg1 : index to i32
        %7 = arith.subi %2, %6 : i32
        %8 = arith.remsi %7, %c100_i32 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst : f64
        %11 = arith.mulf %10, %cst_1 : f64
        affine.for %arg2 = 0 to #map(%arg0) {
          %25 = affine.load %alloc_4[%arg0, %arg2] : memref<200x200xf64>
          %26 = arith.mulf %11, %25 : f64
          %27 = affine.load %alloc[%arg2, %arg1] : memref<200x240xf64>
          %28 = arith.addf %27, %26 : f64
          affine.store %28, %alloc[%arg2, %arg1] : memref<200x240xf64>
          %29 = arith.index_cast %arg2 : index to i32
          %30 = arith.addi %29, %c240_i32 : i32
          %31 = arith.subi %30, %6 : i32
          %32 = arith.remsi %31, %c100_i32 : i32
          %33 = arith.sitofp %32 : i32 to f64
          %34 = arith.divf %33, %cst : f64
          %35 = arith.mulf %34, %25 : f64
          %36 = affine.load %alloca[] : memref<f64>
          %37 = arith.addf %36, %35 : f64
          affine.store %37, %alloca[] : memref<f64>
        }
        %12 = affine.load %alloc[%arg0, %arg1] : memref<200x240xf64>
        %13 = arith.mulf %12, %cst_2 : f64
        %14 = arith.index_cast %arg1 : index to i32
        %15 = arith.subi %4, %14 : i32
        %16 = arith.remsi %15, %c100_i32 : i32
        %17 = arith.sitofp %16 : i32 to f64
        %18 = arith.divf %17, %cst : f64
        %19 = arith.mulf %18, %cst_1 : f64
        %20 = arith.mulf %19, %5 : f64
        %21 = arith.addf %13, %20 : f64
        %22 = affine.load %alloca[] : memref<f64>
        %23 = arith.mulf %22, %cst_1 : f64
        %24 = arith.addf %21, %23 : f64
        affine.store %24, %alloc[%arg0, %arg1] : memref<200x240xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c200 step %c1 {
      scf.for %arg1 = %c0 to %c240 step %c1 {
        %1 = memref.load %alloc[%arg0, %arg1] : memref<200x240xf64>
        %2 = arith.fptosi %1 : f64 to i32
        func.call @print_i32(%2) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<200x240xf64>
    memref.dealloc %alloc_4 : memref<200x200xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
