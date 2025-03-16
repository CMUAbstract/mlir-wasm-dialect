module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2200 = arith.constant 2200 : index
    %c1800 = arith.constant 1800 : index
    %cst = arith.constant 2.200000e+03 : f64
    %cst_0 = arith.constant 1.800000e+03 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1800_i32 = arith.constant 1800 : i32
    %c2200_i32 = arith.constant 2200 : i32
    %alloc = memref.alloc() : memref<1800xf64>
    %alloc_2 = memref.alloc() : memref<2200xf64>
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 1800 {
      affine.store %cst_1, %alloc[%arg0] : memref<1800xf64>
    }
    affine.for %arg0 = 0 to 2200 {
      affine.store %cst_1, %alloc_2[%arg0] : memref<2200xf64>
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.remsi %0, %c2200_i32 : i32
      %2 = arith.sitofp %1 : i32 to f64
      %3 = arith.divf %2, %cst : f64
      affine.for %arg1 = 0 to 1800 {
        %4 = affine.load %alloc[%arg1] : memref<1800xf64>
        %5 = arith.index_cast %arg1 : index to i32
        %6 = arith.addi %5, %c1_i32 : i32
        %7 = arith.muli %0, %6 : i32
        %8 = arith.remsi %7, %c2200_i32 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst : f64
        %11 = arith.mulf %3, %10 : f64
        %12 = arith.addf %4, %11 : f64
        affine.store %12, %alloc[%arg1] : memref<1800xf64>
        %13 = affine.load %alloc_2[%arg0] : memref<2200xf64>
        %14 = arith.remsi %5, %c1800_i32 : i32
        %15 = arith.sitofp %14 : i32 to f64
        %16 = arith.divf %15, %cst_0 : f64
        %17 = arith.mulf %10, %16 : f64
        %18 = arith.addf %13, %17 : f64
        affine.store %18, %alloc_2[%arg0] : memref<2200xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c1800 step %c1 {
      %0 = memref.load %alloc[%arg0] : memref<1800xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    scf.for %arg0 = %c0 to %c2200 step %c1 {
      %0 = memref.load %alloc_2[%arg0] : memref<2200xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<1800xf64>
    memref.dealloc %alloc_2 : memref<2200xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
