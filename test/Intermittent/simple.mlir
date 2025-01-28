// RUN: wasm-opt %s | FileCheck %s


module {
    %a = intermittent.nonvolatile.new() : !intermittent.nonvolatile<i32>
    %b = intermittent.nonvolatile.new() : !intermittent.nonvolatile<i32>

    intermittent.task.idempotent @task1 {
        %x = arith.constant 1 : i32
        %y = arith.constant 2 : i32
        %z = arith.addi %x, %y : i32
        intermittent.task.transition_to @task2  
    }
     intermittent.task.idempotent @task2 {
         intermittent.task.transition_to @task1  
     }
}

