module {
  func.func @bicg_256(%arg0: memref<256x256xf32>, %arg1: memref<256xf32>, %arg2: memref<256xf32>, %arg3: memref<256xf32>, %arg4: memref<256xf32>) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    scf.for %arg5 = %c0 to %c256 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c256_1 = arith.constant 256 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg6 = %c0_0 to %c256_1 step %c1_2 {
        %0 = memref.load %arg1[%arg6] : memref<256xf32>
        %1 = memref.load %arg4[%arg5] : memref<256xf32>
        %2 = memref.load %arg0[%arg5, %arg6] : memref<256x256xf32>
        %3 = arith.mulf %1, %2 : f32
        %4 = arith.addf %0, %3 : f32
        memref.store %4, %arg1[%arg6] : memref<256xf32>
        %5 = memref.load %arg2[%arg5] : memref<256xf32>
        %6 = memref.load %arg3[%arg6] : memref<256xf32>
        %7 = arith.mulf %2, %6 : f32
        %8 = arith.addf %5, %7 : f32
        memref.store %8, %arg2[%arg5] : memref<256xf32>
      }
    }
    return
  }
}

