#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 - d1 - 1)>
module {
  func.func @main() -> i32 {
    %cst = arith.constant -4.010000e+02 : f64
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c400 = arith.constant 400 : index
    %c401_i32 = arith.constant 401 : i32
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<400xf64>
    call @toggle_gpio() : () -> ()
    %alloca = memref.alloca() : memref<f64>
    %0 = llvm.mlir.undef : f64
    affine.store %0, %alloca[] : memref<f64>
    %alloca_2 = memref.alloca() : memref<f64>
    affine.store %0, %alloca_2[] : memref<f64>
    %alloca_3 = memref.alloca() : memref<f64>
    affine.store %0, %alloca_3[] : memref<f64>
    %alloca_4 = memref.alloca() : memref<400xf64>
    affine.store %cst, %alloc[0] : memref<400xf64>
    affine.store %cst_0, %alloca_2[] : memref<f64>
    affine.store %cst, %alloca_3[] : memref<f64>
    affine.for %arg0 = 1 to 400 {
      %1 = affine.load %alloca_3[] : memref<f64>
      %2 = arith.mulf %1, %1 : f64
      %3 = arith.subf %cst_0, %2 : f64
      %4 = affine.load %alloca_2[] : memref<f64>
      %5 = arith.mulf %3, %4 : f64
      affine.store %5, %alloca_2[] : memref<f64>
      affine.store %cst_1, %alloca[] : memref<f64>
      affine.for %arg1 = 0 to #map(%arg0) {
        %13 = affine.apply #map1(%arg0, %arg1)
        %14 = arith.index_cast %13 : index to i32
        %15 = arith.subi %c401_i32, %14 : i32
        %16 = arith.sitofp %15 : i32 to f64
        %17 = affine.load %alloc[%arg1] : memref<400xf64>
        %18 = arith.mulf %16, %17 : f64
        %19 = affine.load %alloca[] : memref<f64>
        %20 = arith.addf %19, %18 : f64
        affine.store %20, %alloca[] : memref<f64>
      }
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.subi %c401_i32, %6 : i32
      %8 = arith.sitofp %7 : i32 to f64
      %9 = affine.load %alloca[] : memref<f64>
      %10 = arith.addf %8, %9 : f64
      %11 = arith.negf %10 : f64
      %12 = arith.divf %11, %5 : f64
      affine.store %12, %alloca_3[] : memref<f64>
      affine.for %arg1 = 0 to #map(%arg0) {
        %13 = affine.load %alloc[%arg1] : memref<400xf64>
        %14 = affine.load %alloc[%arg0 - %arg1 - 1] : memref<400xf64>
        %15 = arith.mulf %12, %14 : f64
        %16 = arith.addf %13, %15 : f64
        affine.store %16, %alloca_4[%arg1] : memref<400xf64>
      }
      affine.for %arg1 = 0 to #map(%arg0) {
        %13 = affine.load %alloca_4[%arg1] : memref<400xf64>
        affine.store %13, %alloc[%arg1] : memref<400xf64>
      }
      affine.store %12, %alloc[%arg0] : memref<400xf64>
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c400 step %c1 {
      %1 = memref.load %alloc[%arg0] : memref<400xf64>
      %2 = arith.fptosi %1 : f64 to i32
      func.call @print_i32(%2) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<400xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio()
  func.func private @print_i32(i32)
}
