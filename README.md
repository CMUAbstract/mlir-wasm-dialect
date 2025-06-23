# WAMI: Compilation to WebAssembly through MLIR without Losing Abstraction


WAMI is an MLIR-based compilation pipeline for WebAssembly. WAMI consists of Wasm dialects explicitly designed to represent high-level Wasm constructs within MLIR. This enables direct generation of high-level Wasm code from corresponding high-level MLIR dialects without losing abstraction, providing a modular and extensible way to incorporate high-level Wasm features.

WAMI is the artifact of [WAMI: Compilation to WebAssembly through MLIR without Losing Abstraction](https://arxiv.org/abs/2506.16048).

## Project Structure

- `include/SsaWasm` and `lib/SsaWasm`: Implement the `SsaWasm` dialect, which represents Wasm in SSA form with explicit operands and results, facilitating the use of standard MLIR analysis and optimization passes.
- `include/Wasm` and `lib/Wasm`: Implement the `Wasm` dialect, which captures Wasm's stack-based semantics with implicit operands and results, optimized for direct emission of Wasm code.
- `include/DCont` and `lib/DCont`: Implement the `dcont` dialect, which models delimited continuations, to be used as an example in the paper.
- `wasm-translate`: Implements the compiler backend, which translates the `Wasm` dialect into Wasm textual format.
- `wasm-opt`: Implements the main driver (counterpart of `mlir-opt`).

The rest of the structure consists of scripts and runtimes for evaluating and testing WAMI:

- `polybench`: Script for performance evaluation on PolyBench.
- `aot-compiler`: Script for AOT compilation for the WAMR runtime.
- `local-executor` and `mcu-wasm-executor`: WAMR setup for execution on local machine and microcontroller, respectively.
- `wasmtime-executor`: Wasmtime setup for execution on local machine

## Building the Project

### LLVM

First, we need to build LLVM with the following options:
```sh
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build . --target check-mlir
```

Now, we can build this project:
```sh
mkdir build && cd build
cmake -G Ninja ..  -DMLIR_DIR=$PREFIX/lib/cmake/mlir  -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit -DCMAKE_BUILD_TYPE=Debug 
cmake --build . --target check-wasm
```

## Testing
We need a few additional tools for testing.

### WABT
Install [WebAssembly Binary Toolkit (WABT)](https://github.com/WebAssembly/wabt)
(either compile it or download a precompiled build)
and add it to `PATH`.
We use `wat2wasm` to convert  `wat` files produced by our lowering passes to
`wasm` binaries.

### WASI-SDK

Install [WASI-SDK](https://github.com/WebAssembly/wasi-sdk) 
to test LLVM-produced Wasm files.
(either compile it or download a precompiled build)
and add it to `PATH`.
We currently use them to link the LLVM-produced Wasm files with the standard library.
We assume that WASI-SDK is installed at `WASI_SDK_PATH`.
For example, this is a part of my `.zshrc`:
```sh
export WASI_SDK_PATu=/Users/byeongje/wasm/wasi-sdk-22.0
```

FIXME: We should change the name of our tool from `wasm-opt` to something else
to avoid conflicts.

### Zephyr

In order to run Wasm files on microcontrollers,
we use [Zephyr](https://docs.zephyrproject.org/latest/index.html).
Install Zephyr following the [guideline](https://docs.zephyrproject.org/latest/develop/getting_started/index.html).
Set the environment variable `ZEPHYR_BASE` to your `zephyrproject/zephyr` directory.
For example, this is a part of my `.zshrc`:
```sh
export ZEPHYR_BASE=/Users/byeongje/zephyrproject/zephyr
```

### WAMR

We use [Wasm Micro Runtime](https://github.com/bytecodealliance/wasm-micro-runtime)
to run Wasm on microcontrollers.
Clone the repository and set the environment variable `WAMR_ROOT_DIR` to point to the directory.
For example, this is a part of my `.zshrc`:
```sh
export WAMR_ROOT_DIR=/Users/byeongje/wasm/wasm-micro-runtime
```

## Usage

This repository contains various tools to compile MLIR files and test them.
We assume that an MLIR file with standard dialects (`arith`, `scf`, and `memref`)
is given.
We have scripts for end-to-end execution as well as each step of it.

### End-to-End Execution

We have a command: `run`, which (1) compiles a given MLIR file to Wasm using
either WAMI or LLVM backend, (2) (optionally) perform
optimizations, (3) (optionally) perform aot compilation, and (4) execution on a MCU.

For example, you can run as follows:

```sh
./run.sh polybench/small/2mm.mlir --compiler=wami --use-aot=false
```

Refer to [mcu-wasm-executor/README.md](mcu-wasm-executor/README.md) to get more
information on testing on MCUs.


### Compile

We can use a script `compile.sh` to compile MLIR files into wasm.
This script supports binaryen optimization, inserting debugging functions, and
etc.
Use `./compile.sh --help` for more options.

```sh
./compile.sh -i test/conv2d.mlir -o conv2d-mlir --compiler=wami
```

For comparison, compilation using LLVM is also supported:
```sh
./compile.sh -i test/conv2d.mlir -o conv2d-llvm --compiler=llvm
```

### AOT Compilation

We have an AOT compiler for faster execution on WAMR.
Refer to [mcu-wasm-executor/aot-compiler/README.md](mcu-wasm-executor/aot-compiler/README.md).

### Execution on MCU

Refer to [mcu-wasm-executor/README.md](mcu-wasm-executor/README.md).

### Execution on wasmtime

Refer to [wasmtime-executor/README.md](wasmtime-executor/README.md).


## Debugging Tips

For debugging wasm code, it is useful to use the `log_i32()` and `log_f32()`
functions (defined in `run-wasm/src/main.rs`). 
These functions can be automatically added by giving `--add-debug-functions`
flag to `compile.sh`.
For debugging wasm files produced by LLVM backend,
import these functions in the WAT file by adding the following:
```
(type (;0;) (func (param i32) (result i32)))
(type (;1;) (func (param f32) (result f32)))
(import "env" "log_i32" (func $log_i32 (type 0)))
(import "env" "log_f32" (func $log_f32 (type 1)))
```
(Reuse function types if possible, and update indices appropriately.)

We can add the following lines at points where we want to read the stack
value:

If the top of the stack is of type `i32`:

```
call $log_i32
```

If the top of the stack is of type `f32`:

```
call $log_f32
```


