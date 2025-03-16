module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4000 = arith.constant 4000 : index
    %cst = arith.constant 4.000000e+03 : f64
    %cst_0 = arith.constant 9.000000e+00 : f64
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant 2.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<4000x4000xf64>
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.sitofp %0 : i32 to f64
      scf.for %arg1 = %c0 to %c4000 step %c1 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.addi %2, %c2_i32 : i32
        %4 = arith.sitofp %3 : i32 to f64
        %5 = arith.mulf %1, %4 : f64
        %6 = arith.addf %5, %cst_1 : f64
        %7 = arith.divf %6, %cst : f64
        memref.store %7, %alloc[%arg0, %arg1] : memref<4000x4000xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 1000 {
      affine.for %arg1 = 1 to 3999 {
        affine.for %arg2 = 1 to 3999 {
          %0 = affine.load %alloc[%arg1 - 1, %arg2 - 1] : memref<4000x4000xf64>
          %1 = affine.load %alloc[%arg1 - 1, %arg2] : memref<4000x4000xf64>
          %2 = arith.addf %0, %1 : f64
          %3 = affine.load %alloc[%arg1 - 1, %arg2 + 1] : memref<4000x4000xf64>
          %4 = arith.addf %2, %3 : f64
          %5 = affine.load %alloc[%arg1, %arg2 - 1] : memref<4000x4000xf64>
          %6 = arith.addf %4, %5 : f64
          %7 = affine.load %alloc[%arg1, %arg2] : memref<4000x4000xf64>
          %8 = arith.addf %6, %7 : f64
          %9 = affine.load %alloc[%arg1, %arg2 + 1] : memref<4000x4000xf64>
          %10 = arith.addf %8, %9 : f64
          %11 = affine.load %alloc[%arg1 + 1, %arg2 - 1] : memref<4000x4000xf64>
          %12 = arith.addf %10, %11 : f64
          %13 = affine.load %alloc[%arg1 + 1, %arg2] : memref<4000x4000xf64>
          %14 = arith.addf %12, %13 : f64
          %15 = affine.load %alloc[%arg1 + 1, %arg2 + 1] : memref<4000x4000xf64>
          %16 = arith.addf %14, %15 : f64
          %17 = arith.divf %16, %cst_0 : f64
          affine.store %17, %alloc[%arg1, %arg2] : memref<4000x4000xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      scf.for %arg1 = %c0 to %c4000 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<4000x4000xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<4000x4000xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
