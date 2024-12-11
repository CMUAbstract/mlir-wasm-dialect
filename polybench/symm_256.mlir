module {
  func.func @main(%arg0: f32, %arg1: f32, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<256x256xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    scf.for %arg5 = %c0 to %c256 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c256_1 = arith.constant 256 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg6 = %c0_0 to %c256_1 step %c1_2 {
        %alloca = memref.alloca() : memref<f32>
        %cst = arith.constant 0.000000e+00 : f32
        memref.store %cst, %alloca[] : memref<f32>
        %c0_3 = arith.constant 0 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg7 = %c0_3 to %arg5 step %c1_4 {
          %10 = memref.load %arg3[%arg5, %arg7] : memref<256x256xf32>
          %11 = memref.load %arg4[%arg5, %arg6] : memref<256x256xf32>
          %12 = memref.load %arg2[%arg7, %arg6] : memref<256x256xf32>
          %13 = arith.mulf %arg0, %11 : f32
          %14 = arith.mulf %10, %13 : f32
          %15 = arith.addf %12, %14 : f32
          memref.store %15, %arg2[%arg7, %arg6] : memref<256x256xf32>
          %16 = memref.load %alloca[] : memref<f32>
          %17 = memref.load %arg4[%arg7, %arg6] : memref<256x256xf32>
          %18 = arith.mulf %10, %17 : f32
          %19 = arith.addf %16, %18 : f32
          memref.store %19, %alloca[] : memref<f32>
        }
        %0 = memref.load %arg3[%arg5, %arg5] : memref<256x256xf32>
        %1 = memref.load %arg4[%arg5, %arg6] : memref<256x256xf32>
        %2 = memref.load %arg2[%arg5, %arg6] : memref<256x256xf32>
        %3 = memref.load %alloca[] : memref<f32>
        %4 = arith.mulf %arg0, %3 : f32
        %5 = arith.mulf %0, %1 : f32
        %6 = arith.mulf %arg0, %5 : f32
        %7 = arith.mulf %arg1, %2 : f32
        %8 = arith.addf %4, %6 : f32
        %9 = arith.addf %7, %8 : f32
        memref.store %9, %arg2[%arg5, %arg6] : memref<256x256xf32>
      }
    }
    return
  }
}

