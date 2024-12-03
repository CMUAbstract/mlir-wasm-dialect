(module
  ;; Import the host-side function 'should_continue_loop'
  (func $should_continue_loop (import "env" "should_continue_loop") (result i32))

  ;; Main function
  (func $main (result i32)
    (local $counter i32)         

    ;; Initialize the counter to 0
    i32.const 0
    local.set $counter

    loop $myloop
      local.get $counter         ;; Get the current counter value
      i32.const 1
      i32.add
      local.set $counter         ;; Update the counter

      call $should_continue_loop ;; Call the host function

      br_if $myloop          
    end                          ;; End of loop

    ;; After the loop, the final counter value is in '$counter'
    local.get $counter           ;; Return the final counter value
  )

  ;; Export the main function
  (export "main" (func $main))
)
