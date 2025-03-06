module {
  func.func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x20x30xf64>, %arg4: memref<?x30xf64>, %arg5: memref<?xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %2 {
      affine.for %arg7 = 0 to %0 {
        affine.for %arg8 = 0 to %1 {
          affine.store %cst, %arg5[%arg8] : memref<?xf64>
          affine.for %arg9 = 0 to %1 {
            %3 = affine.load %arg3[%arg6, %arg7, %arg9] : memref<?x20x30xf64>
            %4 = affine.load %arg4[%arg9, %arg8] : memref<?x30xf64>
            %5 = arith.mulf %3, %4 : f64
            %6 = affine.load %arg5[%arg8] : memref<?xf64>
            %7 = arith.addf %6, %5 : f64
            affine.store %7, %arg5[%arg8] : memref<?xf64>
          }
        }
        affine.for %arg8 = 0 to %1 {
          %3 = affine.load %arg5[%arg8] : memref<?xf64>
          affine.store %3, %arg3[%arg6, %arg7, %arg8] : memref<?x20x30xf64>
        }
      }
    }
    return
  }
  func.func @main() -> i32 {
    %c25 = arith.constant 25 : index
    %c30 = arith.constant 30 : index
    %c20 = arith.constant 20 : index
    %cst = arith.constant 3.000000e+01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c30_i32 = arith.constant 30 : i32
    %alloc = memref.alloc() : memref<25x20x30xf64>
    %alloc_1 = memref.alloc() : memref<30xf64>
    scf.for %arg0 = %c0 to %c25 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c20 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.muli %0, %1 : i32
        scf.for %arg2 = %c0 to %c30 step %c1 {
          %3 = arith.index_cast %arg2 : index to i32
          %4 = arith.addi %2, %3 : i32
          %5 = arith.remsi %4, %c30_i32 : i32
          %6 = arith.sitofp %5 : i32 to f64
          %7 = arith.divf %6, %cst : f64
          memref.store %7, %alloc[%arg0, %arg1, %arg2] : memref<25x20x30xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 25 {
      affine.for %arg1 = 0 to 20 {
        affine.for %arg2 = 0 to 30 {
          affine.store %cst_0, %alloc_1[%arg2] : memref<30xf64>
          %0 = arith.index_cast %arg2 : index to i32
          affine.for %arg3 = 0 to 30 {
            %1 = affine.load %alloc[%arg0, %arg1, %arg3] : memref<25x20x30xf64>
            %2 = arith.index_cast %arg3 : index to i32
            %3 = arith.muli %2, %0 : i32
            %4 = arith.remsi %3, %c30_i32 : i32
            %5 = arith.sitofp %4 : i32 to f64
            %6 = arith.divf %5, %cst : f64
            %7 = arith.mulf %1, %6 : f64
            %8 = affine.load %alloc_1[%arg2] : memref<30xf64>
            %9 = arith.addf %8, %7 : f64
            affine.store %9, %alloc_1[%arg2] : memref<30xf64>
          }
        }
        affine.for %arg2 = 0 to 30 {
          %0 = affine.load %alloc_1[%arg2] : memref<30xf64>
          affine.store %0, %alloc[%arg0, %arg1, %arg2] : memref<25x20x30xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c25 step %c1 {
      scf.for %arg1 = %c0 to %c20 step %c1 {
        scf.for %arg2 = %c0 to %c30 step %c1 {
          %0 = memref.load %alloc[%arg0, %arg1, %arg2] : memref<25x20x30xf64>
          %1 = arith.fptosi %0 : f64 to i32
          func.call @print_i32(%1) : (i32) -> ()
        }
      }
    }
    memref.dealloc %alloc : memref<25x20x30xf64>
    memref.dealloc %alloc_1 : memref<30xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
