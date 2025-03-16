module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2200 = arith.constant 2200 : index
    %cst = arith.constant 9.000000e+03 : f64
    %cst_0 = arith.constant 2.200000e+03 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2200_i32 = arith.constant 2200 : i32
    %alloc = memref.alloc() : memref<2200xf64>
    %alloc_3 = memref.alloc() : memref<1800xf64>
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 2200 {
      affine.store %cst_1, %alloc[%arg0] : memref<2200xf64>
    }
    affine.for %arg0 = 0 to 1800 {
      affine.store %cst_1, %alloc_3[%arg0] : memref<1800xf64>
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 2200 {
        %3 = affine.load %alloc_3[%arg0] : memref<1800xf64>
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.addi %0, %4 : i32
        %6 = arith.remsi %5, %c2200_i32 : i32
        %7 = arith.sitofp %6 : i32 to f64
        %8 = arith.divf %7, %cst : f64
        %9 = arith.sitofp %4 : i32 to f64
        %10 = arith.divf %9, %cst_0 : f64
        %11 = arith.addf %10, %cst_2 : f64
        %12 = arith.mulf %8, %11 : f64
        %13 = arith.addf %3, %12 : f64
        affine.store %13, %alloc_3[%arg0] : memref<1800xf64>
      }
      %1 = arith.index_cast %arg0 : index to i32
      %2 = affine.load %alloc_3[%arg0] : memref<1800xf64>
      affine.for %arg1 = 0 to 2200 {
        %3 = affine.load %alloc[%arg1] : memref<2200xf64>
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.addi %1, %4 : i32
        %6 = arith.remsi %5, %c2200_i32 : i32
        %7 = arith.sitofp %6 : i32 to f64
        %8 = arith.divf %7, %cst : f64
        %9 = arith.mulf %8, %2 : f64
        %10 = arith.addf %3, %9 : f64
        affine.store %10, %alloc[%arg1] : memref<2200xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c2200 step %c1 {
      %0 = memref.load %alloc[%arg0] : memref<2200xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<2200xf64>
    memref.dealloc %alloc_3 : memref<1800xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
