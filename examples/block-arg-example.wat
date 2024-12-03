(module
  ;; Import the host-side function $should_continue_loop
  (func $should_continue_loop (import "env" "should_continue_loop") (result i32))

  ;; Main function
  (func $main (result i32)

    i32.const 0

    loop $myloop (param i32) (result i32)
      i32.const 1 ;; Increment the counter by 1
      i32.add     ;; Updated counter is on top of the stack

      call $should_continue_loop
      br_if $myloop             ;; If $should_continue_loop returns non-zero, branch to $myloop
    end                          ;; End of loop
  )

  ;; Export the main function so it can be called externally
  (export "main" (func $main))
)
