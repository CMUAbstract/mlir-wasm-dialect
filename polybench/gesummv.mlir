module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c2800 = arith.constant 2800 : index
    %cst = arith.constant 2.800000e+03 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.500000e+00 : f64
    %cst_2 = arith.constant 1.200000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c2800_i32 = arith.constant 2800 : i32
    %alloc = memref.alloc() : memref<2800xf64>
    %alloc_3 = memref.alloc() : memref<2800xf64>
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 2800 {
      affine.store %cst_0, %alloc[%arg0] : memref<2800xf64>
      affine.store %cst_0, %alloc_3[%arg0] : memref<2800xf64>
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 2800 {
        %6 = arith.index_cast %arg1 : index to i32
        %7 = arith.muli %0, %6 : i32
        %8 = arith.addi %7, %c1_i32 : i32
        %9 = arith.remsi %8, %c2800_i32 : i32
        %10 = arith.sitofp %9 : i32 to f64
        %11 = arith.divf %10, %cst : f64
        %12 = arith.remsi %6, %c2800_i32 : i32
        %13 = arith.sitofp %12 : i32 to f64
        %14 = arith.divf %13, %cst : f64
        %15 = arith.mulf %11, %14 : f64
        %16 = affine.load %alloc[%arg0] : memref<2800xf64>
        %17 = arith.addf %15, %16 : f64
        affine.store %17, %alloc[%arg0] : memref<2800xf64>
        %18 = arith.addi %7, %c2_i32 : i32
        %19 = arith.remsi %18, %c2800_i32 : i32
        %20 = arith.sitofp %19 : i32 to f64
        %21 = arith.divf %20, %cst : f64
        %22 = arith.mulf %21, %14 : f64
        %23 = affine.load %alloc_3[%arg0] : memref<2800xf64>
        %24 = arith.addf %22, %23 : f64
        affine.store %24, %alloc_3[%arg0] : memref<2800xf64>
      }
      %1 = affine.load %alloc[%arg0] : memref<2800xf64>
      %2 = arith.mulf %1, %cst_1 : f64
      %3 = affine.load %alloc_3[%arg0] : memref<2800xf64>
      %4 = arith.mulf %3, %cst_2 : f64
      %5 = arith.addf %2, %4 : f64
      affine.store %5, %alloc_3[%arg0] : memref<2800xf64>
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c2800 step %c1 {
      %0 = memref.load %alloc_3[%arg0] : memref<2800xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<2800xf64>
    memref.dealloc %alloc_3 : memref<2800xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
