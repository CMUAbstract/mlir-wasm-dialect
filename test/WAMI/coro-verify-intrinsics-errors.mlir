// RUN: wasm-opt %s --coro-verify-intrinsics -split-input-file -verify-diagnostics

// expected-error @+1 {{missing coroutine implementation symbol @coro.impl.missing}}
module {
  func.func private @coro.spawn.missing() -> i64
  func.func private @coro.resume.missing(%h: i64, %x: i32) -> i32

  func.func @main() -> i32 {
    %h = func.call @coro.spawn.missing() : () -> i64
    %x = arith.constant 1 : i32
    %r = func.call @coro.resume.missing(%h, %x) : (i64, i32) -> i32
    return %r : i32
  }
}

// -----

// expected-error @+1 {{coroutine implementation @coro.impl.bad input types must equal spawn args + resume args for kind 'bad'}}
module {
  func.func private @coro.spawn.bad() -> i64
  func.func private @coro.resume.bad(%h: i64, %x: i32) -> i32

  func.func @coro.impl.bad(%x: i64) -> i32 {
    %c = arith.constant 0 : i32
    return %c : i32
  }

  func.func @main() -> i32 {
    %h = func.call @coro.spawn.bad() : () -> i64
    %x = arith.constant 1 : i32
    %r = func.call @coro.resume.bad(%h, %x) : (i64, i32) -> i32
    return %r : i32
  }
}

// -----

module {
  func.func private @coro.is_done.bad(%h: i64) -> i32

  func.func @main() -> i32 {
    %h = arith.constant 0 : i64
    // expected-error @+1 {{coro.is_done.* must have type (i64) -> i1}}
    %d = func.call @coro.is_done.bad(%h) : (i64) -> i32
    return %d : i32
  }
}
