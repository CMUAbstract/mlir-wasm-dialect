func.func @simple_async_example() -> i32 {
  // Spawn an asynchronous task that produces a single integer value.
  %token, %value = async.execute() -> !async.value<i32> {
    %c42 = arith.constant 42 : i32
    async.yield %c42 : i32
  }

  // Await the completion of the asynchronous task.
  %result = async.await %value : !async.value<i32>

  return %result : i32
}