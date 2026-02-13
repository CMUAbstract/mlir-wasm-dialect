#map = affine_map<(d0) -> (d0)>
module {
  func.func @main() -> i32 attributes { exported } {
    %c2000 = arith.constant 2000 : index
    %cst = arith.constant 2.000000e+03 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2000_i32 = arith.constant 2000 : i32
    %alloc = memref.alloc() : memref<2000x2000xf64>
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.addi %0, %c1_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      scf.for %arg1 = %c0 to %2 step %c1 {
        %3 = arith.index_cast %arg1 : index to i32
        %4 = arith.subi %c0_i32, %3 : i32
        %5 = arith.remsi %4, %c2000_i32 : i32
        %6 = arith.sitofp %5 : i32 to f64
        %7 = arith.divf %6, %cst : f64
        %8 = arith.addf %7, %cst_0 : f64
        memref.store %8, %alloc[%arg0, %arg1] : memref<2000x2000xf64>
      }
      scf.for %arg1 = %2 to %c2000 step %c1 {
        memref.store %cst_1, %alloc[%arg0, %arg1] : memref<2000x2000xf64>
      }
      memref.store %cst_0, %alloc[%arg0, %arg0] : memref<2000x2000xf64>
    }
    %alloc_2 = memref.alloc() : memref<2000x2000xf64>
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      scf.for %arg1 = %c0 to %c2000 step %c1 {
        memref.store %cst_1, %alloc_2[%arg0, %arg1] : memref<2000x2000xf64>
      }
    }
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      scf.for %arg1 = %c0 to %c2000 step %c1 {
        %0 = memref.load %alloc[%arg1, %arg0] : memref<2000x2000xf64>
        scf.for %arg2 = %c0 to %c2000 step %c1 {
          %1 = memref.load %alloc[%arg2, %arg0] : memref<2000x2000xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = memref.load %alloc_2[%arg1, %arg2] : memref<2000x2000xf64>
          %4 = arith.addf %3, %2 : f64
          memref.store %4, %alloc_2[%arg1, %arg2] : memref<2000x2000xf64>
        }
      }
    }
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      scf.for %arg1 = %c0 to %c2000 step %c1 {
        %0 = memref.load %alloc_2[%arg0, %arg1] : memref<2000x2000xf64>
        memref.store %0, %alloc[%arg0, %arg1] : memref<2000x2000xf64>
      }
    }
    memref.dealloc %alloc_2 : memref<2000x2000xf64>
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 2000 {
      affine.for %arg1 = 0 to #map(%arg0) {
        affine.for %arg2 = 0 to #map(%arg1) {
          %5 = affine.load %alloc[%arg0, %arg2] : memref<2000x2000xf64>
          %6 = affine.load %alloc[%arg1, %arg2] : memref<2000x2000xf64>
          %7 = arith.mulf %5, %6 : f64
          %8 = affine.load %alloc[%arg0, %arg1] : memref<2000x2000xf64>
          %9 = arith.subf %8, %7 : f64
          affine.store %9, %alloc[%arg0, %arg1] : memref<2000x2000xf64>
        }
        %2 = affine.load %alloc[%arg1, %arg1] : memref<2000x2000xf64>
        %3 = affine.load %alloc[%arg0, %arg1] : memref<2000x2000xf64>
        %4 = arith.divf %3, %2 : f64
        affine.store %4, %alloc[%arg0, %arg1] : memref<2000x2000xf64>
      }
      affine.for %arg1 = 0 to #map(%arg0) {
        %2 = affine.load %alloc[%arg0, %arg1] : memref<2000x2000xf64>
        %3 = arith.mulf %2, %2 : f64
        %4 = affine.load %alloc[%arg0, %arg0] : memref<2000x2000xf64>
        %5 = arith.subf %4, %3 : f64
        affine.store %5, %alloc[%arg0, %arg0] : memref<2000x2000xf64>
      }
      %0 = affine.load %alloc[%arg0, %arg0] : memref<2000x2000xf64>
      %1 = math.sqrt %0 : f64
      affine.store %1, %alloc[%arg0, %arg0] : memref<2000x2000xf64>
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.addi %0, %c1_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      scf.for %arg1 = %c0 to %2 step %c1 {
        %3 = memref.load %alloc[%arg0, %arg1] : memref<2000x2000xf64>
        %4 = arith.fptosi %3 : f64 to i32
        func.call @print_i32(%4) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<2000x2000xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
