#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @main() -> i32 {
    %c200 = arith.constant 200 : index
    %c240 = arith.constant 240 : index
    %cst = arith.constant 2.000000e+02 : f64
    %cst_0 = arith.constant 1.000000e+02 : f64
    %cst_1 = arith.constant 1.000000e+01 : f64
    %cst_2 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c200_i32 = arith.constant 200 : i32
    %alloc = memref.alloc() : memref<200x240xf64>
    %alloc_3 = memref.alloc() : memref<240x240xf64>
    %alloc_4 = memref.alloc() : memref<200x240xf64>
    scf.for %arg0 = %c0 to %c200 step %c1 {
      %1 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c240 step %c1 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.muli %1, %2 : i32
        %4 = arith.remsi %3, %c200_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        %7 = arith.mulf %6, %cst_0 : f64
        %8 = arith.addf %7, %cst_1 : f64
        memref.store %8, %alloc[%arg0, %arg1] : memref<200x240xf64>
        memref.store %cst_2, %alloc_4[%arg0, %arg1] : memref<200x240xf64>
      }
    }
    scf.for %arg0 = %c0 to %c240 step %c1 {
      scf.for %arg1 = %c0 to %c240 step %c1 {
        memref.store %cst_2, %alloc_3[%arg0, %arg1] : memref<240x240xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    %alloca = memref.alloca() : memref<f64>
    %0 = llvm.mlir.undef : f64
    affine.store %0, %alloca[] : memref<f64>
    affine.for %arg0 = 0 to 240 {
      affine.store %cst_2, %alloca[] : memref<f64>
      affine.for %arg1 = 0 to 200 {
        %3 = affine.load %alloc[%arg1, %arg0] : memref<200x240xf64>
        %4 = arith.mulf %3, %3 : f64
        %5 = affine.load %alloca[] : memref<f64>
        %6 = arith.addf %5, %4 : f64
        affine.store %6, %alloca[] : memref<f64>
      }
      %1 = affine.load %alloca[] : memref<f64>
      %2 = math.sqrt %1 : f64
      affine.store %2, %alloc_3[%arg0, %arg0] : memref<240x240xf64>
      affine.for %arg1 = 0 to 200 {
        %3 = affine.load %alloc[%arg1, %arg0] : memref<200x240xf64>
        %4 = arith.divf %3, %2 : f64
        affine.store %4, %alloc_4[%arg1, %arg0] : memref<200x240xf64>
      }
      affine.for %arg1 = #map(%arg0) to 240 {
        affine.store %cst_2, %alloc_3[%arg0, %arg1] : memref<240x240xf64>
        affine.for %arg2 = 0 to 200 {
          %4 = affine.load %alloc_4[%arg2, %arg0] : memref<200x240xf64>
          %5 = affine.load %alloc[%arg2, %arg1] : memref<200x240xf64>
          %6 = arith.mulf %4, %5 : f64
          %7 = affine.load %alloc_3[%arg0, %arg1] : memref<240x240xf64>
          %8 = arith.addf %7, %6 : f64
          affine.store %8, %alloc_3[%arg0, %arg1] : memref<240x240xf64>
        }
        %3 = affine.load %alloc_3[%arg0, %arg1] : memref<240x240xf64>
        affine.for %arg2 = 0 to 200 {
          %4 = affine.load %alloc[%arg2, %arg1] : memref<200x240xf64>
          %5 = affine.load %alloc_4[%arg2, %arg0] : memref<200x240xf64>
          %6 = arith.mulf %5, %3 : f64
          %7 = arith.subf %4, %6 : f64
          affine.store %7, %alloc[%arg2, %arg1] : memref<200x240xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c240 step %c1 {
      scf.for %arg1 = %c0 to %c240 step %c1 {
        %1 = memref.load %alloc_3[%arg0, %arg1] : memref<240x240xf64>
        %2 = arith.fptosi %1 : f64 to i32
        func.call @print_i32(%2) : (i32) -> ()
      }
    }
    scf.for %arg0 = %c0 to %c200 step %c1 {
      scf.for %arg1 = %c0 to %c240 step %c1 {
        %1 = memref.load %alloc_4[%arg0, %arg1] : memref<200x240xf64>
        %2 = arith.fptosi %1 : f64 to i32
        func.call @print_i32(%2) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<200x240xf64>
    memref.dealloc %alloc_3 : memref<240x240xf64>
    memref.dealloc %alloc_4 : memref<200x240xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
