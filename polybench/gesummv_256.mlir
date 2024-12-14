module {
  func.func @main(%arg0: f32, %arg1: f32, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<256xf32>, %arg5: memref<256xf32>, %arg6: memref<256xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    scf.for %arg7 = %c0 to %c256 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c256_1 = arith.constant 256 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg8 = %c0_0 to %c256_1 step %c1_2 {
        %5 = memref.load %arg2[%arg7, %arg8] : memref<256x256xf32>
        %6 = memref.load %arg5[%arg8] : memref<256xf32>
        %7 = memref.load %arg4[%arg7] : memref<256xf32>
        %8 = arith.mulf %5, %6 : f32
        %9 = arith.addf %7, %8 : f32
        memref.store %9, %arg4[%arg7] : memref<256xf32>
        %10 = memref.load %arg3[%arg7, %arg8] : memref<256x256xf32>
        %11 = memref.load %arg5[%arg8] : memref<256xf32>
        %12 = memref.load %arg6[%arg7] : memref<256xf32>
        %13 = arith.mulf %10, %11 : f32
        %14 = arith.addf %12, %13 : f32
        memref.store %14, %arg6[%arg7] : memref<256xf32>
      }
      %0 = memref.load %arg4[%arg7] : memref<256xf32>
      %1 = memref.load %arg6[%arg7] : memref<256xf32>
      %2 = arith.mulf %arg0, %0 : f32
      %3 = arith.mulf %arg1, %1 : f32
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg6[%arg7] : memref<256xf32>
    }
    return
  }
}

