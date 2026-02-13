# Stack-Switching Official Example Corpus (for MLIR Integration Tests)

This catalog was crawled from the official stack-switching proposal clone at:

- `/Users/byeongjee/wasm/stack-switching/proposals/stack-switching/examples`
- `/Users/byeongjee/wasm/stack-switching/proposals/stack-switching/Explainer.md`

Date crawled: 2026-02-13.

## Full Example Inventory

| Source file | Lines | Stack-switching instructions observed |
| --- | ---: | --- |
| actor-lwt.wast | 363 | cont.bind,cont.new,resume,suspend |
| actor.wast | 395 | cont.bind,cont.new,resume,resume_throw,suspend |
| async-await.wast | 335 | cont.bind,cont.new,resume,suspend |
| control-lwt.wast | 356 | cont.bind,cont.new,resume,suspend |
| fun-actor-lwt.wast | 398 | cont.bind,cont.new,resume,suspend |
| fun-lwt.wast | 267 | cont.new,resume,suspend |
| fun-pipes.wast | 88 | cont.bind,cont.new,resume,suspend |
| fun-state.wast | 61 | cont.new,resume,suspend |
| generator-extended.wast | 81 | cont.bind,cont.new,resume,suspend |
| generator.wast | 56 | cont.new,resume,suspend |
| generators.wast | 166 | cont.new,resume,suspend |
| lwt.wast | 294 | cont.new,resume,suspend |
| pipes.wast | 95 | cont.bind,cont.new,resume,suspend |
| scheduler1.wast | 160 | cont.bind,cont.new,resume,suspend |
| scheduler2-throw.wast | 223 | cont.bind,cont.new,resume,resume_throw,suspend,switch |
| scheduler2.wast | 198 | cont.bind,cont.new,resume,suspend,switch |
| static-lwt.wast | 151 | cont.new,resume,suspend |

## Minimal Test Corpus (selected from official examples)

These are the first set to encode as MLIR integration tests because they are both representative and compact.

| Test ID | Official source | Source lines | Coverage intent |
| --- | --- | --- | --- |
| `spec_generator` | `examples/generator.wast` | `17-52` | basic `cont.new + suspend + resume` flow |
| `spec_generator_extended_bind` | `examples/generator-extended.wast` | `24-76` | `cont.bind` adapting continuation arity |
| `spec_fun_pipes` | `examples/fun-pipes.wast` | `11-29`, `53-85` | bidirectional producer/consumer with tagged `suspend/resume` |
| `spec_scheduler2_switch` | `examples/scheduler2.wast` | `82-92`, `176-179` | `resume(... on tag switch)` and explicit `switch` |
| `spec_scheduler2_throw` | `examples/scheduler2-throw.wast` | `85-96`, `107-115`, `199-205` | `resume_throw` and scheduler handoff |

## Notes for this repo

- Current official examples under `examples/` are `switch`-centric; they do not contain `barrier` examples.
- `resume_throw` appears in `scheduler2-throw.wast`.
- `barrier` should be tested via proposal text examples and/or dedicated dialect tests, not this examples folder.

## Added integration seed file

- `test/integration/stack-switching/spec-example-corpus-verify.mlir`

This test file encodes the selected corpus as wasmstack-level verification tests, with each module annotated by source origin.
