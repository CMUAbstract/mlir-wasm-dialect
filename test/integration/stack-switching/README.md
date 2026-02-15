# Stack-Switching Runtime Integration Tests (Wizard, Opt-in)

These tests exercise end-to-end runtime flow for stack-switching and coro
workloads with two benchmark branches:

1. WAMI branch:
   - WAMI/wasmssa lowering
   - `--convert-to-wasmstack`
   - `--verify-wasmstack`
   - `wasm-emit --mlir-to-wasm`
   - Execute generated wasm with Wizard Engine
2. LLVM branch:
   - `coro-to-llvm`
   - MLIR-to-LLVM lowering
   - `mlir-translate --mlir-to-llvmir`
   - `llc` + `wasm-ld`
   - Execute generated wasm with Wizard Engine

The LLVM branch intentionally does not use `wami-convert-*`,
`convert-to-wasmstack`, or `wasm-emit`.

## Environment

Set Wizard location with:

```bash
export WIZARD_ENGINE_DIR=/path/to/wizard-engine
```

Optional overrides:

- `WIZARD_WIZENG_BIN`: explicit wizeng binary path
- `WIZARD_WIZENG_OPTS`: extra wizeng flags

On macOS, `%run_wizard_bin` prefers `bin/wizeng.jvm` to avoid Rosetta issues
with `wizeng.x86-64-darwin`. The runner auto-detects Java from:

- `JAVA_HOME/bin/java`
- `PATH`
- Homebrew OpenJDK (`/opt/homebrew/opt/openjdk/bin/java`, `/usr/local/opt/openjdk/bin/java`)

## Run

```bash
RUN_WIZARD_STACK_SWITCHING=1 \
WIZARD_ENGINE_DIR=/path/to/wizard-engine \
llvm-lit build/test/integration/stack-switching
```

Notes:

- Suite is opt-in and non-gating.
- LLVM runtime branch also requires `llvm_wasm_backend` tools
  (`mlir-translate`, `llc`, `wasm-ld`).

## Printing Debug Values

For runtime trace confidence, examples may import Wizard's `puti` and alias it as
`print_i32` in MLIR:

```mlir
wasmssa.import_func "puti" from "wizeng" as @print_i32 {type = (i32) -> ()}
```

`%run_wizard_bin` enables `--expose=wizeng`, so this host import resolves at runtime.
For LLVM-backend artifacts that import `env.print_i32`, the runner applies a
compatibility rewrite to `wizeng.puti` before execution.

Recommended pattern:

- Keep deterministic runtime-result tests separate from trace tests.
- Runtime-result tests should avoid host print imports when possible.
- Trace tests may keep `print_i32` instrumentation for debugging flow.

Reference test:

- `test/integration/stack-switching/wizard-print-i32-runtime.mlir`
- `test/integration/stack-switching/spec-example-corpus-runtime.mlir`
- `test/integration/stack-switching/wami-stack-switching-runtime.mlir` (result-oriented)
- `test/integration/stack-switching/wami-stack-switching-trace.mlir` (trace-oriented)
- `test/integration/stack-switching/coro-oneshot-runtime.mlir` (coro core one-shot coroutine, dual lowering)
- `test/integration/stack-switching/coro-oneshot-trace-runtime.mlir` (coro core one-shot coroutine with print_i32 trace)
- `test/integration/stack-switching/coro-scheduler-runtime.mlir` (coro core scheduler, dual lowering)
- `test/integration/stack-switching/coro-interruptible-search-runtime.mlir` (coro core interruptible-search, dual lowering)
- `test/integration/stack-switching/coro-multiresume-runtime.mlir` (public API v2 design-only example)

Public API v2 design examples (not runnable in current pipeline yet):

- `docs/examples/coro-public-api-v2/README.md`
- `docs/examples/coro-public-api-v2/coro-generator-v2.mlir`
- `docs/examples/coro-public-api-v2/coro-scheduler-v2.mlir`
- `docs/examples/coro-public-api-v2/coro-interruptible-search-v2.mlir`
- `docs/examples/coro-public-api-v2/coro-multiresume-v2.mlir`

## Viewing Trace Output

To see `print_i32` runtime traces (no `--quiet`), run:

```bash
wasm-opt test/integration/stack-switching/wizard-print-i32-runtime.mlir \
  --convert-to-wasmstack --verify-wasmstack \
| wasm-emit --mlir-to-wasm -o /tmp/wizard-print-i32-runtime.wasm

WIZARD_ENGINE_DIR=/path/to/wizard-engine \
python3 test/integration/stack-switching/run_wizard_bin.py \
  --input /tmp/wizard-print-i32-runtime.wasm --expect-i32 42
```

For the spec-inspired runtime corpus:

```bash
wasm-opt test/integration/stack-switching/spec-example-corpus-runtime.mlir \
  --convert-to-wasmstack --verify-wasmstack \
| wasm-emit --mlir-to-wasm -o /tmp/spec-example-corpus-runtime.wasm
```

Then run it with:

```bash
WIZARD_ENGINE_DIR=/path/to/wizard-engine \
python3 test/integration/stack-switching/run_wizard_bin.py \
  --input /tmp/spec-example-corpus-runtime.wasm --expect-i32 42
```
