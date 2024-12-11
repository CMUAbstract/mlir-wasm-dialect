module {
  func.func @main(%arg0: f32, %arg1: f32, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<256x256xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    scf.for %arg5 = %c0 to %c256 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %0 = arith.addi %arg5, %c1_1 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg6 = %c0_0 to %0 step %c1_2 {
        %1 = memref.load %arg2[%arg5, %arg6] : memref<256x256xf32>
        %2 = arith.mulf %arg1, %1 : f32
        memref.store %2, %arg2[%arg5, %arg6] : memref<256x256xf32>
        %c0_3 = arith.constant 0 : index
        %c256_4 = arith.constant 256 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg7 = %c0_3 to %c256_4 step %c1_5 {
          %3 = memref.load %arg3[%arg5, %arg7] : memref<256x256xf32>
          %4 = memref.load %arg4[%arg6, %arg7] : memref<256x256xf32>
          %5 = memref.load %arg4[%arg5, %arg7] : memref<256x256xf32>
          %6 = memref.load %arg3[%arg6, %arg7] : memref<256x256xf32>
          %7 = memref.load %arg2[%arg5, %arg6] : memref<256x256xf32>
          %8 = arith.mulf %3, %4 : f32
          %9 = arith.mulf %5, %6 : f32
          %10 = arith.addf %8, %9 : f32
          %11 = arith.mulf %arg0, %10 : f32
          %12 = arith.addf %7, %11 : f32
          memref.store %12, %arg2[%arg5, %arg6] : memref<256x256xf32>
        }
      }
    }
    return
  }
}

