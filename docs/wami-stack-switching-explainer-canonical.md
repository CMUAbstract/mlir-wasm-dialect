# Canonical WAMI Mapping for Stack-Switching Explainer Snippets

Date: 2026-02-13

## Purpose

This document defines canonical WAMI/wasmssa representations for the
stack-switching program snippets in:

`/Users/byeongjee/wasm/stack-switching/proposals/stack-switching/Explainer.md`

This version is intentionally aligned with the stack-switching proposal shape:
`resume` carries handler label dispatch metadata only; handler code is not
embedded in `resume` regions.

## Scope

Covered snippets:

1. `(tag $gen (param i32))`
2. `module $generator` skeleton
3. basic `func $generator`
4. basic `func $consumer`
5. `module $scheduler1` skeleton
6. `module $scheduler2` skeleton
7. `(tag $gen (param i32) (result i32))`
8. extended `func $generator`
9. `$ft0/$ft1/$ct0/$ct1` type block
10. extended `func $consumer` using `cont.bind`
11. `func $schedule_task` using `resume_throw`

## Canonical Design Rules

1. `wami.resume` and `wami.resume_throw` do not own handler regions.
2. Handlers are encoded as dispatch entries on the op itself.
3. Handler code lives in enclosing structured CFG (`wasmssa.block`/`wasmssa.loop`).
4. Ordinary handlers are label targets (`on tag label`).
5. Switch handlers are switch entries (`on tag switch`) with no label body.
6. Label targets are represented as structured-label depth (`level`) to match
Wasm label-index semantics.
7. Level resolution is lexical at the `resume` site: `level = 0` means the
innermost enclosing structured label, `level = 1` the next enclosing label,
and so on outward.
8. For `on_label`, transferred values are ordered as:
   tag payload values (in tag parameter order), then captured continuation.

## Canonical `resume` Shape

```mlir
"wami.resume"(%cont, %args...) <{
  cont_type = @ct,
  handlers = [
    #wami.on_label<tag = @t0, level = 0>,
    #wami.on_label<tag = @t1, level = 1>,
    #wami.on_switch<tag = @s0>
  ]
}> : (!wami.cont<@ct>, arg_types...) -> result_types...
```

Interpretation:

- `on_label` means: if resumed continuation suspends with `tag`, branch to the
  structured label at `level` (`0` = innermost), passing
  `(payload..., continuation)` in that order.
- `on_switch` means: install switch handler for `tag` (no label body).

## Per-Snippet Canonical Mapping

### 1) `(tag $gen (param i32))`

```mlir
module {
  wami.tag @gen : (i32) -> ()
}
```

### 2) `module $generator` skeleton

```mlir
module {
  wami.type.func @ft = () -> ()
  wami.type.cont @ct = cont @ft

  wasmssa.import_func "print_i32" from "spectest" as @print_i32 {type = (i32) -> ()}
  wami.tag @gen : (i32) -> ()

  wasmssa.func @generator() { ... }
  wasmssa.func @consumer() { ... }
}
```

### 3) Basic `func $generator`

```mlir
wasmssa.func @generator() {
  %c100 = wasmssa.const 100 : i32
  %c1 = wasmssa.const 1 : i32
  %c0 = wasmssa.const 0 : i32

  wasmssa.loop(%c100) : i32 : {
  ^loop(%i: i32):
    "wami.suspend"(%i) <{tag = @gen}> : (i32) -> ()
    %next = wasmssa.sub %i %c1 : i32
    %keep = wasmssa.ne %next %c0 : i32
    wasmssa.branch_if %keep to level 0 with args(%next : i32) else ^done
  ^done:
    wasmssa.return
  }> ^exit

^exit(%_ignored: i32):
  wasmssa.return
}
```

### 4) Basic `func $consumer` (full canonical translation)

```mlir
wasmssa.func @consumer() {
  %f = wami.ref.func @generator : !wami.funcref<@generator>
  %c0 = wami.cont.new %f : !wami.funcref<@generator> as @ct -> !wami.cont<@ct>

  wasmssa.loop(%c0) : !wami.cont<@ct> : {
  ^loop(%c: !wami.cont<@ct>):
    // Models:
    //   (block $on_gen (result i32 (ref $ct))
    //     (resume $ct (on $gen $on_gen) (local.get $c))
    //     (return))
    wasmssa.block() : (i32, !wami.cont<@ct>) : {
    ^resume_site:
      "wami.resume"(%c) <{
        cont_type = @ct,
        handlers = [#wami.on_label<tag = @gen, level = 0>]
      }> : (!wami.cont<@ct>) -> ()
      // Normal return from resumed continuation.
      wasmssa.return
    }> ^on_gen

  ^on_gen(%value: i32, %k: !wami.cont<@ct>):
    wasmssa.call @print_i32(%value) : (i32) -> ()
    // Continue loop with captured continuation.
    wasmssa.block_return %k : !wami.cont<@ct>
  }> ^after_loop

^after_loop(%_unused: !wami.cont<@ct>):
  wasmssa.return
}
```

### 5) `module $scheduler1` skeleton (asymmetric scheduling)

```mlir
module {
  wami.type.func @ft = () -> ()
  wami.type.cont @ct = cont @ft
  wami.tag @yield : () -> ()

  wasmssa.func @entry(%initial_task: !wami.funcref<@ft>) {
    ...
    wasmssa.loop() : () : {
    ^resume_next:
      %next = wasmssa.call @task_dequeue_non_null : () -> !wami.cont<@ct>
      ...
      wasmssa.block() : (!wami.cont<@ct>) : {
      ^resume_site:
        "wami.resume"(%next) <{
          cont_type = @ct,
          handlers = [#wami.on_label<tag = @yield, level = 0>]
        }> : (!wami.cont<@ct>) -> ()
        // task finished
        wasmssa.block_return
      }> ^on_yield

    ^on_yield(%k: !wami.cont<@ct>):
      wasmssa.call @task_enqueue(%k) : (!wami.cont<@ct>) -> ()
      wasmssa.block_return
    }>
  }

  wasmssa.func @task_0() {
    ...
    "wami.suspend"() <{tag = @yield}> : () -> ()
    ...
    wasmssa.return
  }
}
```

### 6) `module $scheduler2` skeleton (switch-based scheduling)

```mlir
module {
  // Explainer recursion:
  //   (type $ft (func (param (ref null $ct))))
  //   (type $ct (cont $ft))
  wami.type.func @ft = (!wami.cont<@ct>?) -> ()
  wami.type.cont @ct = cont @ft
  wami.tag @yield : () -> ()

  wasmssa.func @entry(%initial_task: !wami.funcref<@ft>) {
    ...
    wasmssa.loop() : () : {
    ^resume_next:
      %next = wasmssa.call @task_dequeue : () -> !wami.cont<@ct>?
      %null = wami.ref.null : !wami.cont<@ct>?
      "wami.resume"(%next, %null) <{
        cont_type = @ct,
        handlers = [#wami.on_switch<tag = @yield>]
      }> : (!wami.cont<@ct>, !wami.cont<@ct>?) -> ()
      // task finished execution
      wasmssa.block_return
    }>
  }

  wasmssa.func @yield_to_next() -> !wami.cont<@ct>? {
    %next = wasmssa.call @task_dequeue : () -> !wami.cont<@ct>?
    ...
    %from_switch = "wami.switch"(%next) <{cont_type = @ct, tag = @yield}>
      : (!wami.cont<@ct>) -> !wami.cont<@ct>?
    wasmssa.return %from_switch : !wami.cont<@ct>?
  }
}
```

### 7) `(tag $gen (param i32) (result i32))`

```mlir
module {
  wami.tag @gen : (i32) -> i32
}
```

### 8) Extended `func $generator` (`suspend` has result)

```mlir
wasmssa.func @generator() {
  %c100 = wasmssa.const 100 : i32
  %c1 = wasmssa.const 1 : i32

  wasmssa.loop(%c100) : i32 : {
  ^loop(%i: i32):
    %reset = "wami.suspend"(%i) <{tag = @gen}> : (i32) -> i32
    %next = wasmssa.if %reset -> i32 {
      %v100 = wasmssa.const 100 : i32
      wasmssa.yield %v100 : i32
    } else {
      %dec = wasmssa.sub %i %c1 : i32
      wasmssa.yield %dec : i32
    }
    %keep = wasmssa.ne %next %c1 : i32
    wasmssa.branch_if %keep to level 0 with args(%next : i32) else ^done
  ^done:
    wasmssa.return
  }> ^exit

^exit(%_ignored: i32):
  wasmssa.return
}
```

### 9) `$ft0/$ft1/$ct0/$ct1` type block

```mlir
module {
  wami.type.func @ft0 = () -> ()
  wami.type.func @ft1 = (i32) -> ()
  wami.type.cont @ct0 = cont @ft0
  wami.type.cont @ct1 = cont @ft1
}
```

### 10) Extended `func $consumer` (`cont.bind`)

```mlir
wasmssa.func @consumer(%iter_ref: !wasmssa<local ref to i32>) {
  %f = wami.ref.func @generator : !wami.funcref<@generator>
  %c0 = wami.cont.new %f : !wami.funcref<@generator> as @ct0 -> !wami.cont<@ct0>
  %iter0 = wasmssa.const 1 : i32
  %c42 = wasmssa.const 42 : i32
  %c1 = wasmssa.const 1 : i32

  wasmssa.loop(%c0, %iter0) : (!wami.cont<@ct0>, i32) : {
  ^loop(%cur0: !wami.cont<@ct0>, %iter: i32):
    // Explainer-style ambient local state for iteration count.
    "wasmssa.local_set"(%iter_ref, %iter)
      : (!wasmssa<local ref to i32>, i32) -> ()
    wasmssa.block() : (i32, !wami.cont<@ct1>) : {
    ^resume_site:
      "wami.resume"(%cur0) <{
        cont_type = @ct0,
        handlers = [#wami.on_label<tag = @gen, level = 0>]
      }> : (!wami.cont<@ct0>) -> ()
      wasmssa.return
    }> ^on_gen

  ^on_gen(%yielded: i32, %cur1: !wami.cont<@ct1>):
    wasmssa.call @print_i32(%yielded) : (i32) -> ()
    %iter_cur = wasmssa.local_get %iter_ref : !wasmssa<local ref to i32>
    %flag = wasmssa.eq %iter_cur %c42 : i32
    %next0 = wami.cont.bind %cur1, %flag
      : !wami.cont<@ct1>, i32 as (@ct1 -> @ct0) -> !wami.cont<@ct0>
    %iter_next = wasmssa.add %iter_cur %c1 : i32
    wasmssa.block_return %next0, %iter_next : !wami.cont<@ct0>, i32
  }> ^done

^done(%_c: !wami.cont<@ct0>, %_iter: i32):
  wasmssa.return
}
```

### 11) `func $schedule_task` (`resume_throw`)

```mlir
wasmssa.func @schedule_task(%c: !wami.cont<@ct>?) {
  %count = wasmssa.call @task_queue_count : () -> i32
  %limit = wasmssa.global_get @concurrent_task_limit : i32
  %full = wasmssa.ge_si %count %limit : i32 -> i32

  wasmssa.if %full {
    %victim = wasmssa.call @task_dequeue : () -> !wami.cont<@ct>
    wasmssa.block {
    ^exc_handler:
      wasmssa.try_table (catch @abort ^exc_handler) {
        "wami.resume_throw"(%victim) <{
          cont_type = @ct,
          exception_tag = @abort,
          handlers = []
        }> : (!wami.cont<@ct>) -> ()
      }
      wasmssa.block_return
    }
  }

  wasmssa.call @task_enqueue(%c) : (!wami.cont<@ct>?) -> ()
  wasmssa.return
}
```

## Instruction Mapping (Informative)

| Explainer form | Canonical WAMI form |
| --- | --- |
| `cont.new $ct (ref.func $f)` | `wami.ref.func` + `wami.cont.new` |
| `resume $ct (on $t $l) ...` | `wami.resume` + `#wami.on_label<tag=@t, level=...>` |
| `resume $ct (on $t switch) ...` | `wami.resume` + `#wami.on_switch<tag=@t>` |
| `suspend $t ...` | `wami.suspend` |
| `cont.bind ...` | `wami.cont.bind` |
| `resume_throw ...` | `wami.resume_throw` |
| `switch $ct $t` | `wami.switch` |

## Design Note

This document is normative for desired IR shape. It is intentionally separate
from current implementation status tracked in:

`docs/stack-switching-missing-issues.md`
