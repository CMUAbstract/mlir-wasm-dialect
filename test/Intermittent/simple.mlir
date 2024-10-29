// RUN: wasm-opt %s | FileCheck %s

module {
    intermittent.task.idempotent @task1 {
        %x = intermittent.nonvolatile.new() : !intermittent.nonvolatile<i32>
        %y = arith.constant 1 : i32
        intermittent.nonvolatile.store %y, %x : !intermittent.nonvolatile<i32> 
        intermittent.task.transition_to @task2 (%x : !intermittent.nonvolatile<i32>)
    }
     intermittent.task.idempotent @task2 {
         %x = intermittent.nonvolatile.new() : !intermittent.nonvolatile<i32>
         %y = intermittent.nonvolatile.load %x : !intermittent.nonvolatile<i32>
         intermittent.nonvolatile.store %y, %x : !intermittent.nonvolatile<i32>
         intermittent.task.transition_to 
     }
}

