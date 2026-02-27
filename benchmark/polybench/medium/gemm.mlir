module {
  func.func @main() -> i32 attributes { exported } {
    %c0 = arith.constant 0 : index
    %c200 = arith.constant 200 : index
    %c220 = arith.constant 220 : index
    %cst = arith.constant 2.200000e+02 : f64
    %cst_0 = arith.constant 2.400000e+02 : f64
    %cst_1 = arith.constant 2.000000e+02 : f64
    %cst_2 = arith.constant 1.500000e+00 : f64
    %cst_3 = arith.constant 1.200000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c240_i32 = arith.constant 240 : i32
    %c220_i32 = arith.constant 220 : i32
    %c200_i32 = arith.constant 200 : i32
    %alloc = memref.alloc() : memref<200x220xf64>
    scf.for %arg0 = %c0 to %c200 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c220 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.muli %0, %1 : i32
        %3 = arith.addi %2, %c1_i32 : i32
        %4 = arith.remsi %3, %c200_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst_1 : f64
        memref.store %6, %alloc[%arg0, %arg1] : memref<200x220xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 200 {
      affine.for %arg1 = 0 to 220 {
        %1 = affine.load %alloc[%arg0, %arg1] : memref<200x220xf64>
        %2 = arith.mulf %1, %cst_3 : f64
        affine.store %2, %alloc[%arg0, %arg1] : memref<200x220xf64>
      }
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 240 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.addi %1, %c1_i32 : i32
        %3 = arith.muli %0, %2 : i32
        %4 = arith.remsi %3, %c240_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst_0 : f64
        %7 = arith.mulf %6, %cst_2 : f64
        affine.for %arg2 = 0 to 220 {
          %8 = arith.index_cast %arg2 : index to i32
          %9 = arith.addi %8, %c2_i32 : i32
          %10 = arith.muli %1, %9 : i32
          %11 = arith.remsi %10, %c220_i32 : i32
          %12 = arith.sitofp %11 : i32 to f64
          %13 = arith.divf %12, %cst : f64
          %14 = arith.mulf %7, %13 : f64
          %15 = affine.load %alloc[%arg0, %arg2] : memref<200x220xf64>
          %16 = arith.addf %15, %14 : f64
          affine.store %16, %alloc[%arg0, %arg2] : memref<200x220xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c200 step %c1 {
      scf.for %arg1 = %c0 to %c220 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<200x220xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<200x220xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
