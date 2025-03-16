module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c2000 = arith.constant 2000 : index
    %c2300 = arith.constant 2300 : index
    %cst = arith.constant 2.300000e+03 : f64
    %cst_0 = arith.constant 2.600000e+03 : f64
    %cst_1 = arith.constant 2.000000e+03 : f64
    %cst_2 = arith.constant 1.500000e+00 : f64
    %cst_3 = arith.constant 1.200000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c2600_i32 = arith.constant 2600 : i32
    %c2300_i32 = arith.constant 2300 : i32
    %c2000_i32 = arith.constant 2000 : i32
    %alloc = memref.alloc() : memref<2000x2300xf64>
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c2300 step %c1 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.muli %0, %1 : i32
        %3 = arith.addi %2, %c1_i32 : i32
        %4 = arith.remsi %3, %c2000_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst_1 : f64
        memref.store %6, %alloc[%arg0, %arg1] : memref<2000x2300xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 2000 {
      affine.for %arg1 = 0 to 2300 {
        %1 = affine.load %alloc[%arg0, %arg1] : memref<2000x2300xf64>
        %2 = arith.mulf %1, %cst_3 : f64
        affine.store %2, %alloc[%arg0, %arg1] : memref<2000x2300xf64>
      }
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 2600 {
        %1 = arith.index_cast %arg1 : index to i32
        %2 = arith.addi %1, %c1_i32 : i32
        %3 = arith.muli %0, %2 : i32
        %4 = arith.remsi %3, %c2600_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst_0 : f64
        %7 = arith.mulf %6, %cst_2 : f64
        affine.for %arg2 = 0 to 2300 {
          %8 = arith.index_cast %arg2 : index to i32
          %9 = arith.addi %8, %c2_i32 : i32
          %10 = arith.muli %1, %9 : i32
          %11 = arith.remsi %10, %c2300_i32 : i32
          %12 = arith.sitofp %11 : i32 to f64
          %13 = arith.divf %12, %cst : f64
          %14 = arith.mulf %7, %13 : f64
          %15 = affine.load %alloc[%arg0, %arg2] : memref<2000x2300xf64>
          %16 = arith.addf %15, %14 : f64
          affine.store %16, %alloc[%arg0, %arg2] : memref<2000x2300xf64>
        }
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c2000 step %c1 {
      scf.for %arg1 = %c0 to %c2300 step %c1 {
        %0 = memref.load %alloc[%arg0, %arg1] : memref<2000x2300xf64>
        %1 = arith.fptosi %0 : f64 to i32
        func.call @print_i32(%1) : (i32) -> ()
      }
    }
    memref.dealloc %alloc : memref<2000x2300xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
