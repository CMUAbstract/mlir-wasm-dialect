// RUN: wasm-opt %s | FileCheck %s


module {
    %a = intermittent.nonvolatile.new() : !intermittent.nonvolatile<i32>
    %b = intermittent.nonvolatile.new() : !intermittent.nonvolatile<i32>

    intermittent.task.idempotent @task1 {
        %a2 = intermittent.nonvolatile.load %a : !intermittent.nonvolatile<i32>
        %x = arith.constant 1 : i32
        %y = arith.addi %x, %a2 : i32
        intermittent.nonvolatile.store %y, %a : !intermittent.nonvolatile<i32>
        intermittent.task.transition_to @task2 (%a : !intermittent.nonvolatile<i32>) 
    }
     intermittent.task.idempotent @task2 {
         %a2 = intermittent.nonvolatile.load %a : !intermittent.nonvolatile<i32>
         intermittent.nonvolatile.store %a2, %b : !intermittent.nonvolatile<i32>
         intermittent.task.transition_to 
     }
}

