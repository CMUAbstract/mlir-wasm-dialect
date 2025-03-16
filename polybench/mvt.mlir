module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4000 = arith.constant 4000 : index
    %cst = arith.constant 4.000000e+03 : f64
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c4000_i32 = arith.constant 4000 : i32
    %alloc = memref.alloc() : memref<4000xf64>
    %alloc_0 = memref.alloc() : memref<4000xf64>
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %0 = arith.index_cast %arg0 : index to i32
      %1 = arith.remsi %0, %c4000_i32 : i32
      %2 = arith.sitofp %1 : i32 to f64
      %3 = arith.divf %2, %cst : f64
      memref.store %3, %alloc[%arg0] : memref<4000xf64>
      %4 = arith.addi %0, %c1_i32 : i32
      %5 = arith.remsi %4, %c4000_i32 : i32
      %6 = arith.sitofp %5 : i32 to f64
      %7 = arith.divf %6, %cst : f64
      memref.store %7, %alloc_0[%arg0] : memref<4000xf64>
    }
    call @toggle_gpio() : () -> ()
    affine.for %arg0 = 0 to 4000 {
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 4000 {
        %1 = affine.load %alloc[%arg0] : memref<4000xf64>
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.muli %0, %2 : i32
        %4 = arith.remsi %3, %c4000_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        %7 = arith.addi %2, %c3_i32 : i32
        %8 = arith.remsi %7, %c4000_i32 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst : f64
        %11 = arith.mulf %6, %10 : f64
        %12 = arith.addf %1, %11 : f64
        affine.store %12, %alloc[%arg0] : memref<4000xf64>
      }
    }
    affine.for %arg0 = 0 to 4000 {
      %0 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 4000 {
        %1 = affine.load %alloc_0[%arg0] : memref<4000xf64>
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.muli %2, %0 : i32
        %4 = arith.remsi %3, %c4000_i32 : i32
        %5 = arith.sitofp %4 : i32 to f64
        %6 = arith.divf %5, %cst : f64
        %7 = arith.addi %2, %c4_i32 : i32
        %8 = arith.remsi %7, %c4000_i32 : i32
        %9 = arith.sitofp %8 : i32 to f64
        %10 = arith.divf %9, %cst : f64
        %11 = arith.mulf %6, %10 : f64
        %12 = arith.addf %1, %11 : f64
        affine.store %12, %alloc_0[%arg0] : memref<4000xf64>
      }
    }
    call @toggle_gpio() : () -> ()
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %0 = memref.load %alloc[%arg0] : memref<4000xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    scf.for %arg0 = %c0 to %c4000 step %c1 {
      %0 = memref.load %alloc_0[%arg0] : memref<4000xf64>
      %1 = arith.fptosi %0 : f64 to i32
      func.call @print_i32(%1) : (i32) -> ()
    }
    memref.dealloc %alloc : memref<4000xf64>
    memref.dealloc %alloc_0 : memref<4000xf64>
    return %c0_i32 : i32
  }
  func.func private @toggle_gpio() attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @print_i32(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}
