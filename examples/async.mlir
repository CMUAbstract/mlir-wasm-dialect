module {
  async.func @task1() -> !async.token {
    %c42 = arith.constant 42 : i32
    %token = call @task2() : () -> !async.token
    async.await %token : !async.token
    async.return
  }
  async.func @task2() -> !async.token {
    %c42 = arith.constant 42 : i32
    async.return
  }
  func.func @main() {
    // Spawn an asynchronous task that produces a single integer value.
    %token = async.call @task1() : () -> !async.token

    // Await the completion of the asynchronous task.
    async.await %token : !async.token 

    return 
  }
}