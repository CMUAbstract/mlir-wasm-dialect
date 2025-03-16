#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c2000 = arith.constant 2000 : index
    %c2600 = arith.constant 2600 : index
    %cst = arith.constant 2.000000e+03 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.500000e+00 : f64
    %cst_2 = arith.constant 1.200000e+00 : f64
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant -9.990000e+02 : f64
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c2600_i32 = arith.constant 2600 : i32
    %alloc = memref.alloc() : memref<2000x2600xf64>
    %alloc_4 = memref.alloc() : memref<2000x2000xf64>
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      %1 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c2600 step %c1 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.addi %1, %2 : i32
        %4 = arith.remsi %3, %c100_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        memref.store %6, %alloc[%arg0, %arg1] : memref<2000x2600xf64>
      }
    }
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.addi %1, %c1_i32 : i32
      %3 = arith.index_cast %2 : i32 to index
      scf.for %arg1 = %c0 to %3 step %c1 {
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.addi %1, %4 : i32
        %6 = arith.remsi %5, %c100_i32 : i32
        %7 = arith.sitofp %6 : i32 to f64
        %8 = arith.divf %7, %cst : f64
        memref.store %8, %alloc_4[%arg0, %arg1] : memref<2000x2000xf64>
      }
      scf.for %arg1 = %3 to %c2000 step %c1 {
        memref.store %cst_3, %alloc_4[%arg0, %arg1] : memref<2000x2000xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    %alloca = memref.alloca() : memref<f64>
    %0 = llvm.mlir.undef : f64
    affine.store %0, %alloca[] : memref<f64>
    affine.for %arg0 = 0 to 2000 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.addi %1, %c2600_i32 : i32
      %3 = arith.index_cast %arg0 : index to i32
      %4 = arith.addi %3, %c2600_i32 : i32
      %5 = affine.load %alloc_4[%arg0, %arg0] : memref<2000x2000xf64>
      affine.for %arg1 = 0 to 2600 {
        affine.store %cst_0, %alloca[] : memref<f64>
        %6 = arith.index_cast %arg1 : index to i32
        %7 = arith.subi %2, %6 : i32
        %8 = arith.remsi %7, %c100_i32 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst : f64
        %11 = arith.mulf %10, %cst_1 : f64
        affine.for %arg2 = 0 to #map(%arg0) {
          %25 = affine.load %alloc_4[%arg0, %arg2] : memref<2000x2000xf64>
          %26 = arith.mulf %11, %25 : f64
          %27 = affine.load %alloc[%arg2, %arg1] : memref<2000x2600xf64>
          %28 = arith.addf %27, %26 : f64
          affine.store %28, %alloc[%arg2, %arg1] : memref<2000x2600xf64>
          %29 = arith.index_cast %arg2 : index to i32
          %30 = arith.addi %29, %c2600_i32 : i32
          %31 = arith.subi %30, %6 : i32
          %32 = arith.remsi %31, %c100_i32 : i32
          %33 = arith.sitofp %32 : i32 to f64
          %34 = arith.divf %33, %cst : f64
          %35 = arith.mulf %34, %25 : f64
          %36 = affine.load %alloca[] : memref<f64>
          %37 = arith.addf %36, %35 : f64
          affine.store %37, %alloca[] : memref<f64>
        }
        %12 = affine.load %alloc[%arg0, %arg1] : memref<2000x2600xf64>
        %13 = arith.mulf %12, %cst_2 : f64
        %14 = arith.index_cast %arg1 : index to i32
        %15 = arith.subi %4, %14 : i32
        %16 = arith.remsi %15, %c100_i32 : i32
        %17 = arith.sitofp %16 : i32 to f64
        %18 = arith.divf %17, %cst : f64
        %19 = arith.mulf %18, %cst_1 : f64
        %20 = arith.mulf %19, %5 : f64
        %21 = arith.addf %13, %20 : f64
        %22 = affine.load %alloca[] : memref<f64>
        %23 = arith.mulf %22, %cst_1 : f64
        %24 = arith.addf %21, %23 : f64
        affine.store %24, %alloc[%arg0, %arg1] : memref<2000x2600xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      scf.for %arg1 = %c0 to %c2600 step %c1 {
        %1 = memref.load %alloc[%arg0, %arg1] : memref<2000x2600xf64>
        %2 = arith.fptosi %1 : f64 to i32
        func.call @print_i32(%2) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<2000x2600xf64>
    memref.dealloc %alloc_4 : memref<2000x2000xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
