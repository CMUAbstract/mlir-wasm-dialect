# Stack-Switching Runtime Integration Tests (Wizard, Opt-in)

These tests exercise end-to-end runtime flow for stack-switching workloads:

1. WAMI/wasmssa input IR
2. `--convert-to-wasmstack`
3. `--verify-wasmstack`
4. `wasm-emit --mlir-to-wasm`
5. Execute generated wasm with Wizard Engine

## Environment

Set Wizard location with:

```bash
export WIZARD_ENGINE_DIR=/path/to/wizard-engine
```

Optional overrides:

- `WIZARD_WIZENG_BIN`: explicit wizeng binary path
- `WIZARD_WIZENG_OPTS`: extra wizeng flags

## Run

```bash
RUN_WIZARD_STACK_SWITCHING=1 \
WIZARD_ENGINE_DIR=/path/to/wizard-engine \
llvm-lit build/test/integration/stack-switching
```

Notes:

- Suite is opt-in and non-gating.
- Some tests are intentionally `XFAIL` while stack-switching wasm emission is still incomplete.
