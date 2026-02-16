# WAMI: Compilation to WebAssembly through MLIR without Losing Abstraction

WAMI is an MLIR-based compilation pipeline for WebAssembly. It is the artifact
of [WAMI: Compilation to WebAssembly through MLIR without Losing
Abstraction](https://arxiv.org/abs/2506.16048).

## Branch Status

This branch is under active reimplementation.

- Active dialect/passes:
  - `wasmssa` (upstream MLIR dialect)
  - `wami` (`include/WAMI/`, `lib/WAMI/`)
  - `wasmstack` (`include/wasmstack/`, `lib/wasmstack/`)
- Removed legacy dialects/passes:
  - `ssawasm`, `wasm`, `intermittent`

Preferred lowering path for new development:

1. Standard MLIR (`arith`, `math`, `func`, `scf`, `memref`)
2. `--wami-convert-all` (or explicit `--wami-convert-*` passes, including
   `--wami-convert-math`)
3. `--reconcile-unrealized-casts`
4. `--convert-to-wasmstack`
5. `--verify-wasmstack`
6. `wasm-emit --mlir-to-wasm`

## Project Layout

- `include/WAMI`, `lib/WAMI`: WAMI dialect + conversion passes
- `include/wasmstack`, `lib/wasmstack`: WasmStack dialect, stackification,
  verifier, and emitter bridge
- `include/Target/WasmStack`, `lib/Target/WasmStack`: WebAssembly binary emitter
- `wasm-opt`: driver for passes
- `wasm-emit`: MLIR-to-Wasm binary tool
- `polybench`: benchmark scripts
- `local-executor`, `mcu-wasm-executor`, `wasmtime-executor`: runtime harnesses

## Build

Build LLVM/MLIR first, then this project:

```sh
mkdir -p build
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=$PREFIX/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target check-wasm
```

## Test

Run focused suites:

```sh
llvm-lit build/test/WAMI
llvm-lit build/test/wasmstack
```

Run one test file:

```sh
llvm-lit build/test/wasmstack/full-pipeline-verify.mlir
```

Opt-in execution benchmarks:

```sh
RUN_WASMTIME_BENCH=1 llvm-lit build/test/integration/benchmarks
```

Opt-in stack-switching runtime tests with Wizard Engine:

```sh
RUN_WIZARD_STACK_SWITCHING=1 \
WIZARD_ENGINE_DIR=/path/to/wizard-engine \
llvm-lit build/test/integration/stack-switching
```

## Tool Prerequisites

- [WABT](https://github.com/WebAssembly/wabt) (`wasm-validate`, `wasm-objdump`,
  `wat2wasm`)
- [WASI-SDK](https://github.com/WebAssembly/wasi-sdk) for LLVM backend script
  flow (`WASI_SDK_PATH`)
- [Zephyr](https://docs.zephyrproject.org/latest/index.html) and
  [WAMR](https://github.com/bytecodealliance/wasm-micro-runtime) for MCU runs

Example environment variables:

```sh
export WASI_SDK_PATH=/path/to/wasi-sdk
export ZEPHYR_BASE=/path/to/zephyrproject/zephyr
export WAMR_ROOT_DIR=/path/to/wasm-micro-runtime
export WIZARD_ENGINE_DIR=/path/to/wizard-engine
```

## Usage

### Preferred New Pipeline

```sh
build/bin/wasm-opt input.mlir \
  --wami-convert-all \
  --reconcile-unrealized-casts \
  --convert-to-wasmstack \
  --verify-wasmstack \
  -o out.wasmstack.mlir

build/bin/wasm-emit out.wasmstack.mlir --mlir-to-wasm -o out.wasm
```

### Existing Scripted Flow

`compile.sh` and `run.sh` are used by existing benchmark/runtime scripts:

```sh
./compile.sh -i test/conv2d.mlir -o conv2d-wami --compiler=wami
./compile.sh -i test/conv2d.mlir -o conv2d-llvm --compiler=llvm
./run.sh polybench/small/2mm.mlir --compiler=wami --use-aot=false
```

`compile.sh --compiler=wami` follows the same `wami -> wasmstack -> wasm-emit`
pipeline shown above.
