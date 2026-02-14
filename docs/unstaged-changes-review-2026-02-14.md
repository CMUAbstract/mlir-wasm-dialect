# Review: Unstaged Changes Concerns

Date: 2026-02-14
Scope: current unstaged stack-switching / wasm-emit changes

## Findings (ordered by severity)

### 1. High: `wasm-emit` can assert-crash on unresolved `wasmstack.ref.func`

`OpcodeEmitter` emits `wasmstack.ref.func` by calling `IndexSpace::getFuncIndex`,
which asserts when the symbol is missing, instead of producing a diagnostic.
`verify-wasmstack` does not currently validate `ref.func` symbol existence, so
malformed input can reach the emitter and abort.

- Code refs:
  - `lib/Target/WasmStack/OpcodeEmitter.cpp:1001`
  - `lib/Target/WasmStack/IndexSpace.cpp:179`
  - `lib/wasmstack/VerifyWasmStack.cpp:344`
  - `include/wasmstack/WasmStackOps.td:1308`

- Reproducer used:
  - create module with `wasmstack.ref.func @missing`
  - run `build/bin/wasm-opt <file> --verify-wasmstack`
  - run `build/bin/wasm-emit --mlir-to-wasm <verified-file> -o /tmp/out.wasm`
  - observed assertion: `function not found in index space`

### 2. Medium: new stack-switching immediate lookups also assert on malformed symbols

New emitter paths for `cont_type` / `tag` use assert-based lookup helpers.
On unverified malformed input, `wasm-emit` aborts instead of returning a user
error.

- Code refs:
  - `lib/Target/WasmStack/OpcodeEmitter.cpp:1028`
  - `lib/Target/WasmStack/OpcodeEmitter.cpp:1034`
  - `lib/Target/WasmStack/OpcodeEmitter.cpp:1043`
  - `lib/Target/WasmStack/IndexSpace.cpp:205`
  - `lib/Target/WasmStack/IndexSpace.cpp:219`

- Reproducer used:
  - create module with `wasmstack.suspend @missing`
  - run `build/bin/wasm-emit --mlir-to-wasm <file> -o /tmp/out.wasm`
  - observed assertion: `tag not found in index space`

### 3. Low: missing negative tests for unresolved-symbol robustness

Added/updated tests cover happy-path stack-switching emission, but there is no
negative test for unresolved `ref.func` / stack-switching symbol immediates
causing graceful diagnostics.

- Code ref:
  - `test/wasmstack/emit-binary-stack-switching.mlir:1`

## Assumption

This review assumes `wasm-emit` should remain robust when invoked directly,
without requiring `--verify-wasmstack` beforehand.
