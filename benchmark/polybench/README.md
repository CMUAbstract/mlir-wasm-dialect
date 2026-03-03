# PolyBench Benchmark Suite on Apollo4 Blue Plus

This directory provides instructions for testing an MLIR-based port of the
PolyBench benchmark suite on the Apollo4 Blue Plus device. It allows
configuration of compilers, optimization levels, and evaluation setups.

## Key Features
- Supports `--compiler=wami` and `--compiler=llvm` through `../../toolchain/run.sh`
- Configurable LLVM optimization levels (`-O0`, `-O1`, ...)
- Configurable Binaryen optimization levels (`-O0`, `-O1`, ...)
- Configurable AOT optimization levels (`0`, `1`, `2`, `3`)

## Prerequisites

### Toolchain Setup (macOS)

1. **Zephyr RTOS** (required for MCU deployment):
   ```sh
   # Create workspace and install west
   mkdir -p ~/zephyrproject
   uv venv ~/zephyrproject/.venv
   source ~/zephyrproject/.venv/bin/activate
   uv pip install west

   # Initialize and fetch all modules
   cd ~/zephyrproject
   west init
   west update  # takes several minutes

   # Install Zephyr Python dependencies
   uv pip install -r ~/zephyrproject/zephyr/scripts/requirements.txt
   ```

2. **GNU ARM Embedded Toolchain** (cross-compiler for Cortex-M4):
   ```sh
   brew install gcc-arm-embedded
   ```

3. **WAMR** (WebAssembly Micro Runtime, compiled as part of the Zephyr build):
   ```sh
   git clone https://github.com/bytecodealliance/wasm-micro-runtime.git ~/wasm/wasm-micro-runtime
   ```

4. **J-Link** (for flashing the board):
   ```sh
   brew install --cask segger-jlink
   ```

5. **Environment variables** (add to your `.envrc` or shell profile):
   ```sh
   export ZEPHYRPROJECT=~/zephyrproject
   export ZEPHYR_BASE=~/zephyrproject/zephyr
   export ZEPHYR_TOOLCHAIN_VARIANT=gnuarmemb
   export GNUARMEMB_TOOLCHAIN_PATH=/Applications/ArmGNUToolchain/<version>/arm-none-eabi
   export WAMR_ROOT_DIR=~/wasm/wasm-micro-runtime
   ```
   Replace `<version>` with the installed toolchain version (e.g., `15.2.rel1`).

### Hardware Setup

1. Connect the Apollo4 Blue Plus board and Saleae Logic device to the host machine.
2. Connect to pin GND and 22 (see the uses of `am_hal_gpi_state_write` in `./src/*.h`)
3. Install the Logic2 software: https://www.saleae.com/pages/downloads
4. In Logic2, open Settings and enable the Automation Server feature.

## Running the Benchmarks
1. Setup virtualenv
   - It is recommended to install [uv](https://docs.astral.sh/uv/) and run `uv sync` in this directory. Then prefix commands with `uv run`, or activate the venv with `source .venv/bin/activate`.
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
- `--compiler=wami` uses the WAMI → WasmStack pipeline
  (`--wami-convert-all --convert-to-wasmstack --verify-wasmstack` + `wasm-emit`)
  via `compile.sh`.

## Caution
Please manually verify that the Wasm files run successfully without errors (for
example, by using minicom), and confirm that you are measuring intended behavior
before collecting performance numbers.

## Small-Only Correctness Validation

### WAMI-only (quick smoke test)

Compile and run every `polybench/small` benchmark with WAMI only. No LLVM
toolchain or `WASI_SDK_PATH` needed:

```sh
./validate_small_correctness.py --wami-only
```

### Differential test (WAMI vs LLVM)

Compile each benchmark with both `--compiler=wami` and `--compiler=llvm`, run
both through `wasmtime-executor` in `--print-mode=hash`, and compare `actual`
return value, `print_count`, and `print_hash`:

```sh
./validate_small_correctness.py
```

### Useful options

```sh
./validate_small_correctness.py --wami-only --filter floyd-warshall
./validate_small_correctness.py --keep-temp
./validate_small_correctness.py --binaryen-opt-flags=-O2 --llvm-opt-level=O3
```

### Requirements

- `cargo +nightly` must be available for building `wasmtime-executor`.
- `WASI_SDK_PATH` must be set for the differential test (not needed with
  `--wami-only`).
