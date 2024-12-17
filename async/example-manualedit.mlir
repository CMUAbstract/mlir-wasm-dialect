module {
  func.func @task1() -> !async.token attributes {passthrough = ["presplitcoroutine"]} {
    %0 = async.runtime.create : !async.token
    %1 = async.coro.id
    %2 = async.coro.begin %1
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(42 : i32) : i32
    %4 = call @task2() : () -> !async.token
    %5 = async.coro.save %2
    async.runtime.await_and_resume %4, %2 : !async.token
    async.coro.suspend %5, ^bb7, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %6 = async.runtime.is_error %4 : !async.token
    llvm.cond_br %6, ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    async.runtime.set_available %0 : !async.token
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    async.runtime.set_error %0 : !async.token
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    async.coro.free %1, %2
    llvm.br ^bb7
  ^bb6:  // pred: ^bb1
    async.coro.free %1, %2
    llvm.br ^bb7
  ^bb7:  // 3 preds: ^bb1, ^bb5, ^bb6
    async.coro.end %2
    return %0 : !async.token
  }
  func.func @task2() -> !async.token attributes {passthrough = ["presplitcoroutine"]} {
    %0 = async.runtime.create : !async.token
    %1 = async.coro.id
    %2 = async.coro.begin %1
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(42 : i32) : i32
    async.runtime.set_available %0 : !async.token
    llvm.br ^bb2
  ^bb2:  // pred: ^bb1
    async.coro.free %1, %2
    llvm.br ^bb4
  ^bb3:  // no predecessors
    async.coro.free %1, %2
    cf.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    async.coro.end %2
    return %0 : !async.token
  }
  llvm.func @main() {
    %0 = func.call @task1() : () -> !async.token
    async.runtime.await %0 : !async.token
    %1 = async.runtime.is_error %0 : !async.token
    %2 = llvm.mlir.constant(true) : i1
    %3 = llvm.xor %1, %2 : i1
    llvm.cond_br %3, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.return
  ^bb2:  // pred: ^bb0
    llvm.unreachable
  }
}

