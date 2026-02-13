# PolyBench Benchmark Suite on Apollo4 Blue Plus

This directory provides instructions for testing an MLIR-based port of the
PolyBench benchmark suite on the Apollo4 Blue Plus device. It allows 
configuration of compilers, optimization levels, and evaluation setups.

## Key Features
- Supports `--compiler=wami` and `--compiler=llvm` through `../run.sh`
- Configurable LLVM optimization levels (`-O0`, `-O1`, ...)
- Configurable Binaryen optimization levels (`-O0`, `-O1`, ...)
- Configurable AOT optimization levels (`0`, `1`, `2`, `3`)

## Prerequisites
1. Connect the Apollo4 Blue Plus board and Saleae Logic device to the host machine.
2. Connect to pin GND and 22 (see the uses of `am_hal_gpi_state_write` in `./src/*.h`)
3. Install the Logic2 software: https://www.saleae.com/pages/downloads
4. In Logic2, open Settings and enable the Automation Server feature.

## Running the Benchmarks
1. Setup virtualenv
   - It is recommended to install pipenv and run `pipenv install; pipenv shell` in this directory.
2. Execute the command below to view generated benchmark commands:
```sh
./gencmds.py
```
3. To run specific benchmarks, pipe the output of gencmds.py through jq to filter by tag or compiler, then pipe that into runcmds.py. For example:
```sh
./gencmds.py | jq -c 'select(.tag == "atax" and .compiler == "wami" and .binaryen_opt_level == 4)' | ./runcmds.py
```
4. To run all tests at once:
```sh
./gencmds.py | ./runcmds.py
```

## Notes on Compiler Modes
- `--compiler=llvm` is the LLVM-based comparison baseline.
- `--compiler=wami` follows the repository WAMI compilation path used by
  `run.sh`/`compile.sh`.
- If you need strict evaluation of the newest `wami -> wasmstack` pipeline,
  compile with `wasm-opt ... --convert-to-wasmstack --verify-wasmstack` and
  `wasm-emit` directly, then run the resulting `.wasm` with the target runtime.

## Caution
Please manually verify that the Wasm files run successfully without errors (for
example, by using minicom), and confirm that you are measuring intended behavior
before collecting performance numbers.

## Small-Only Correctness Check (WAMI vs LLVM)

You can run a differential correctness sweep for `polybench/small` with:

```sh
./validate_small_correctness.py
```

What this does:
- Compiles each `polybench/small/*.mlir` with both `--compiler=wami` and
  `--compiler=llvm`.
- Runs both outputs with `wasmtime-executor` in `--print-mode=hash`.
- Compares `actual` return value, `print_count`, and `print_hash`.

Useful options:

```sh
./validate_small_correctness.py --filter floyd-warshall
./validate_small_correctness.py --keep-temp
./validate_small_correctness.py --binaryen-opt-flags=-O2 --llvm-opt-flags=-O3
```

Requirements:
- `WASI_SDK_PATH` must be set (LLVM flow in `compile.sh`).
- `cargo +nightly` must be available for building `wasmtime-executor`.
