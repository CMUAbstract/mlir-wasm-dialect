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
- Some tests are intentionally `XFAIL` while stack-switching wasm emission is still incomplete.

## Printing Debug Values

For runtime trace confidence, examples may import Wizard's `puti` and alias it as
`print_i32` in MLIR:

```mlir
wasmssa.import_func "puti" from "wizeng" as @print_i32 {type = (i32) -> ()}
```

`%run_wizard_bin` enables `--expose=wizeng`, so this host import resolves at runtime.

Reference test:

- `test/integration/stack-switching/wizard-print-i32-runtime.mlir`
- `test/integration/stack-switching/spec-example-corpus-runtime.mlir` (`XFAIL` for now)

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

This now emits wasm and clears `ref.func` declaration checks, but runtime
execution still fails in Wizard due to continuation-reference typing mismatch
(Wizard expects typed `(ref null cont #N)`-style values and reports type
mismatch against generic `contref`/stack values).
