// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --wami-convert-all --reconcile-unrealized-casts --coro-to-wami --convert-to-wasmstack --verify-wasmstack -split-input-file
// RUN: wasm-opt %s --coro-verify-intrinsics --coro-normalize --coro-to-llvm -split-input-file | mlir-opt --lower-affine --convert-scf-to-cf --convert-arith-to-llvm="index-bitwidth=32" --convert-func-to-llvm="index-bitwidth=32" --memref-expand --expand-strided-metadata --finalize-memref-to-llvm="index-bitwidth=32" --convert-cf-to-llvm="index-bitwidth=32" --convert-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o /dev/null

module {
  func.func private @coro.spawn.oneshot() -> i64
  func.func private @coro.resume.oneshot(%h: i64, %x: i32)
      -> (i64, i1, i32)

  func.func @coro.impl.oneshot(%x: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %r = arith.addi %x, %c1 : i32
    return %r : i32
  }

  func.func @main() -> i32 attributes { exported } {
    %h = func.call @coro.spawn.oneshot() : () -> i64
    %x = arith.constant 41 : i32
    %h2, %done, %r = func.call @coro.resume.oneshot(%h, %x)
        : (i64, i32) -> (i64, i1, i32)
    return %r : i32
  }
}

// -----

module {
  func.func private @coro.spawn.multiresume() -> i64
  func.func private @coro.resume.multiresume(%h: i64, %x: i32)
      -> (i64, i1, i32)

  func.func @coro.impl.multiresume(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %r = arith.muli %x, %c2 : i32
    return %r : i32
  }

  func.func @main() -> i32 attributes { exported } {
    %h = func.call @coro.spawn.multiresume() : () -> i64
    %a = arith.constant 10 : i32
    %b = arith.constant 11 : i32
    %h1, %d1, %r1 = func.call @coro.resume.multiresume(%h, %a)
        : (i64, i32) -> (i64, i1, i32)
    %h2, %d2, %r2 = func.call @coro.resume.multiresume(%h1, %b)
        : (i64, i32) -> (i64, i1, i32)
    %sum = arith.addi %r1, %r2 : i32
    return %sum : i32
  }
}

// -----

module {
  func.func private @coro.spawn.task() -> i64
  func.func private @coro.resume.task(%h: i64, %x: i32)
      -> (i64, i1, i32)

  func.func @coro.impl.task(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %r = arith.muli %x, %c2 : i32
    return %r : i32
  }

  func.func @main() -> i32 attributes { exported } {
    %h1 = func.call @coro.spawn.task() : () -> i64
    %h2 = func.call @coro.spawn.task() : () -> i64

    %a = arith.constant 10 : i32
    %b = arith.constant 11 : i32
    %h1n, %d1, %r1 = func.call @coro.resume.task(%h1, %a)
        : (i64, i32) -> (i64, i1, i32)
    %h2n, %d2, %r2 = func.call @coro.resume.task(%h2, %b)
        : (i64, i32) -> (i64, i1, i32)
    %sum = arith.addi %r1, %r2 : i32
    return %sum : i32
  }
}

// -----

module {
  func.func private @coro.spawn.search() -> i64
  func.func private @coro.resume.search(%h: i64, %candidate: i32)
      -> (i64, i1, i32)

  func.func @coro.impl.search(%candidate: i32) -> i32 {
    %threshold = arith.constant 42 : i32
    %zero = arith.constant 0 : i32
    %is_valid = arith.cmpi eq, %candidate, %threshold : i32
    %r = scf.if %is_valid -> i32 {
      scf.yield %candidate : i32
    } else {
      scf.yield %zero : i32
    }
    return %r : i32
  }

  func.func @main() -> i32 attributes { exported } {
    %h = func.call @coro.spawn.search() : () -> i64
    %probe = arith.constant 42 : i32
    %h2, %done, %r = func.call @coro.resume.search(%h, %probe)
        : (i64, i32) -> (i64, i1, i32)
    return %r : i32
  }
}
