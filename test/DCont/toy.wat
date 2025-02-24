(module
(type $ft (func))
(type $ct (cont $ft))
(import "env" "print_i32" (func $print_i32 (param i32)))
(import "env" "print_i32_2" (func $print_i32_2 (param i32)))
(global $a (mut i32) (i32.const 0))
(global $b (mut i32) (i32.const 100))
(tag $yield)


(func $task1
    (local $i i32)
    (block $b1
        (loop $l1
            (local.get $i)
            (i32.const 100)
            i32.ge_u 
            br_if $b1

            (global.get $a)
            (call $print_i32)

            (global.get $a)
            (i32.const 1)
            (i32.add)
            (global.set $a)

            (suspend $yield)

            ;; returned from suspend
            (local.get $i)
            (i32.const 1)
            (i32.add)
            (local.set $i)
            br $l1
        )
    )
return)
(elem declare func $task1)

(func $task2
    (local $i i32)
    (block $b1
        (loop $l1
            (local.get $i)
            (i32.const 100)
            i32.ge_u 
            br_if $b1

            (global.get $b)
            (call $print_i32)

            (global.get $b)
            (i32.const 1)
            (i32.add)
            (global.set $b)

            (suspend $yield)

            ;; returned from suspend
            (local.get $i)
            (i32.const 1)
            (i32.add)
            (local.set $i)
            br $l1
        )
    )
return)
(elem declare func $task2)

(func $main
    (local $i i32)
    (local $task1_handle (ref null $ct))
    (local $task2_handle (ref null $ct))

    (ref.func $task1)
    (cont.new $ct)
    (local.set $task1_handle)

    (ref.func $task2)
    (cont.new $ct)
    (local.set $task2_handle)

    (block $b1
        (loop $l1
            (local.get $i)
            (i32.const 200)
            i32.ge_u 
            br_if $b1

            (local.get $i)
            (i32.const 2)
            (i32.rem_u)
            (if 
                (then
                    (block $on_yield (result (ref null $ct))
                        (local.get $task1_handle)
                        (resume $ct (on $yield $on_yield))
                        (ref.null $ct)
                    )
                    local.set $task1_handle
                )
                (else
                    (block $on_yield (result (ref null $ct))
                        (local.get $task2_handle)
                        (resume $ct (on $yield $on_yield))
                        (ref.null $ct)
                    )
                    local.set $task2_handle
                )
            ) 

            local.get $i
            i32.const 1
            i32.add
            local.set $i

            br $l1
        )
    ) 
return)

(export "main" (func $main))
)

