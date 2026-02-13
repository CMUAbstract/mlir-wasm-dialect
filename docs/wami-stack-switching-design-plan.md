# WAMI Stack Switching Design Plan (Canonical Label-Dispatch)

Date: 2026-02-13
Status: Active implementation plan

## Summary

Migrate WAMI stack-switching to the canonical model from
`docs/wami-stack-switching-explainer-canonical.md`.

Core contract:

1. `wami.resume` / `wami.resume_throw` are metadata-only dispatch ops.
2. Handler bodies are represented by surrounding structured CFG (`wasmssa.block`, `wasmssa.loop`), not by regions on `resume`.
3. `on_label` dispatch uses lexical depth (`level`) with Wasm label-index semantics.

Migration mode: hard switch now (no dual-form compatibility).

## Superseded Design Mapping

1. `handler_tags + handlers region` on `wami.resume`
- Superseded by: `handlers` array containing typed handler attrs.

2. `wami.handler.yield`
- Superseded by: structured control flow transfer to enclosing labels.

3. Region-local handler typing
- Superseded by: symbol/type validation of handler attrs and structured-label depth checks.

## Public IR/API Changes

## 1) WAMI handler attrs

Introduce typed handler attrs:

1. `#wami.on_label<tag = @t, level = N>`
2. `#wami.on_switch<tag = @t>`

## 2) `wami.resume` / `wami.resume_throw`

1. Replace `handler_tags` with `handlers`.
2. Remove handler region.
3. Keep continuation signature checks and `resume_throw` no-normal-result rule.

## 3) Remove region-only helper op

1. Remove `wami.handler.yield` (no longer part of canonical IR).

## 4) WasmStack coordination

1. Lower `on_label(level)` to concrete wasmstack handler label targets.
2. Lower `on_switch` to wasmstack switch-compatible handler form.
3. Validate handler tag + label correctness in wasmstack verification.

## Implementation Sequence

1. Docs refresh
- Rewrite `docs/stack-switching-missing-issues.md` and this file.

2. WAMI attr + op shape migration
- Add typed handler attrs.
- Change op definitions and parser/printer/verifier expectations.

3. WAMI verifier migration
- Remove region-based checks.
- Enforce handler attr validity, unique tags, and `level >= 0`.

4. WAMI -> wasmstack lowering migration
- Consume `handlers` attrs only.
- Resolve `level` against enclosing structured labels.
- Add missing barrier emission in dispatcher.

5. wasmstack verification hardening
- Check handler target labels exist in active control context.
- Keep symbol/type checks.

6. Test migration
- Update WAMI, wasmstack, and integration tests away from region handlers.
- Add malformed-handler negative cases for canonical attrs.

## Testing Matrix

## WAMI tests

1. Parse/print of canonical handler attrs.
2. Verifier negatives:
- duplicate handler tag
- invalid handler attr type
- negative `level`
- unknown tag symbol

## WasmStack tests

1. Lowering checks from WAMI canonical handlers.
2. Verifier negatives for unknown handler labels / malformed handlers.

## Integration tests

1. Canonical `resume`/`resume_throw` forms in runtime corpus.
2. Keep binary-emission-dependent cases `XFAIL` until backend completion.

## Risks and Mitigations

1. Risk: incorrect `level` resolution in nested control flow.
- Mitigation: deep nesting tests with known label-depth mapping.

2. Risk: hard switch breaks legacy tests abruptly.
- Mitigation: migrate all stack-switching tests in same patch set.

3. Risk: mismatch between WAMI canonical handlers and wasmstack representation.
- Mitigation: explicit lowering checks and verifier diagnostics for each handler kind.

## Assumptions

1. Local llvm-project changes enabling wasmssa ops with WAMI reference types are available in the toolchain.
2. Canonical explainer document is the source of truth for IR shape.
