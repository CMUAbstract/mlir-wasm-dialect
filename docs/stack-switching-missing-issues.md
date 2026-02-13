# Stack Switching: Remaining Gaps After Canonical Label-Dispatch Design

Date: 2026-02-13
Status: Active backlog for canonical model in `docs/wami-stack-switching-explainer-canonical.md`

## Context

This file replaces the earlier region-handler review.
The canonical design is now:

1. `wami.resume` / `wami.resume_throw` contain handler metadata only.
2. Handler code lives in surrounding `wasmssa.block` / `wasmssa.loop` control flow.
3. `on_label` uses structured label depth (`level`), with `level=0` as innermost.

## Superseded / Stale Items from Region-Handler Era

1. `resume` / `resume_throw` handler regions needing lowering
- Old status: treated as missing executable lowering for region bodies.
- New status: stale by design. Canonical IR does not use handler regions.

2. `wami.handler.yield` semantics
- Old status: central to handler region typing.
- New status: stale by design. Removed from canonical representation.

## Current Blocking Issues

## P0: IR Shape and Verification

1. WAMI handler attr model is not implemented
- Needed: typed handler clauses (`#wami.on_label`, `#wami.on_switch`) and `handlers` attr on `resume` / `resume_throw`.
- Impact: canonical snippets cannot be represented directly.

2. WAMI verifier still validates `handler_tags + regions`
- Needed: verifier migration to metadata-only handlers.
- Impact: canonical form is rejected.

3. Emitter path still consumes handler regions
- Needed: `WAMI -> wasmstack` lowering from handler attrs only.
- Impact: canonical form cannot lower.

## P1: Lowering Correctness and WasmStack Verification

4. Label-depth (`level`) resolution is not implemented end-to-end
- Needed: resolve lexical structured-label depth to concrete wasmstack handler targets during lowering.
- Impact: wrong/non-deterministic control transfer for non-local flow.

5. WasmStack verification does not fully validate handler target labels
- Needed: ensure handler label exists in active structured control context and is shape-compatible.
- Impact: malformed handlers can pass verification.

6. `wami.barrier` emission path is incomplete in stackification emitter
- Needed: explicit emission hook and stack-effect handling.
- Impact: valid canonical programs with barrier fail lowering.

## P2: Binary Emission and Runtime Completion

7. Stack-switching opcode emission in `wasm-emit` is incomplete
- Missing coverage includes stack-switching instructions and refs (`ref.func`, `ref.null`, `cont.new`, `cont.bind`, `resume`, `resume_throw`, `suspend`, `barrier`, `switch`).
- Impact: runtime integration tests remain `XFAIL`.

8. Continuation/tag section/index emission is incomplete
- Needed: symbol indexing and binary section serialization for stack-switching declarations.
- Impact: emitted wasm is structurally incomplete.

9. Type encoding and error propagation are incomplete for non-numeric stack-switching types
- Impact: failures are late or non-diagnostic in wasm binary emission.

## Priority Order

1. P0: canonical IR and verifier migration.
2. P1: deterministic lowering and target validation.
3. P2: binary emission and runtime un-XFAIL.

## Acceptance Checklist

1. Canonical snippets in `docs/wami-stack-switching-explainer-canonical.md` parse and verify in WAMI.
2. `--convert-to-wasmstack --verify-wasmstack` succeeds for canonical stack-switching coverage.
3. Handler-target validation catches malformed label-depth cases.
4. Known binary-emission gaps are explicit in tests/docs until implemented.
