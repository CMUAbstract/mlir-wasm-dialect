module {
  func.func @doitgen_256(%arg0: memref<256x256x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256xf32>) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c256 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c256_1 = arith.constant 256 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg4 = %c0_0 to %c256_1 step %c1_2 {
        %c0_3 = arith.constant 0 : index
        %c256_4 = arith.constant 256 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg5 = %c0_3 to %c256_4 step %c1_5 {
          %cst = arith.constant 0.000000e+00 : f32
          memref.store %cst, %arg2[%arg5] : memref<256xf32>
          %c0_9 = arith.constant 0 : index
          %c256_10 = arith.constant 256 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg6 = %c0_9 to %c256_10 step %c1_11 {
            %0 = memref.load %arg0[%arg3, %arg4, %arg6] : memref<256x256x256xf32>
            %1 = memref.load %arg1[%arg6, %arg5] : memref<256x256xf32>
            %2 = memref.load %arg2[%arg5] : memref<256xf32>
            %3 = arith.mulf %0, %1 : f32
            %4 = arith.addf %2, %3 : f32
            memref.store %4, %arg2[%arg5] : memref<256xf32>
          }
        }
        %c0_6 = arith.constant 0 : index
        %c256_7 = arith.constant 256 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg5 = %c0_6 to %c256_7 step %c1_8 {
          %0 = memref.load %arg2[%arg5] : memref<256xf32>
          memref.store %0, %arg0[%arg3, %arg4, %arg5] : memref<256x256x256xf32>
        }
      }
    }
    return
  }
}

