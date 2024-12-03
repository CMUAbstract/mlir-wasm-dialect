# MLIR WebAssembly Dialect

## Prerequisites

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
export WASI_SDK_PATH=/Users/byeongje/wasm/wasi-sdk-22.0
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

We have a command: `run-aot`, which (1) compiles a given MLIR file to Wasm using
either `--convert-to-wasm` pass or LLVM backend, (2) (optionally) perform
optimizations, (3) (optionally) perform aot compilation, and (4) execution on a MCU.

For example, you can run as follows:

```sh
./run-aot.sh test/lenet.mlir --compiler=llvm --testcase="MNIST_LLVM" --binaryen-opt-flags="-O3" --use-aot=true \
-- --opt-level=0 --target=thumbv7em --target-abi=eabihf --cpu=cortex-m4
```

Refer to [mcu-wasm-executor/README.md](mcu-wasm-executor/README.md) to get more
information on testing on MCUs.


### Compile

We can use a script `compile.sh` to compile MLIR files into wasm.
This script supports binaryen optimization, inserting debugging functions, and
etc.
Use `./compile.sh --help` for more options.

```sh
./compile.sh -i test/conv2d.mlir -o conv2d-mlir --compiler=mlir
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


## Generating Inputs

To use the `--convert-to-wasm` pass, we require that an MLIR file only contains
the standard dialects (arith, scf, and memref). Any higher-level
dialects must be lowered to these standard dialects beforehand.

Currently, we have two examples: `test/conv2d.mlir` and `test/lenet.mlir`.
We outline the creation of `test/conv2d.mlir` here; `test/lenet.mlir` was generated
similarly.

We start from `test/conv2d-tosa.mlir`, which is generated from TFLite, and apply
the following pipelines to produce `test/conv2d.mlir`.

```sh
mlir-opt test/conv2d-tosa.mlir \
--pass-pipeline="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, \
canonicalize, tosa-infer-shapes, tosa-optional-decompositions, \
tosa-layerwise-constant-fold, tosa-make-broadcastable, tosa-to-arith, \
tosa-to-tensor), convert-tensor-to-linalg)" -o \
test/conv2d-linalg.mlir
# We had to split this into two commands because currently the
# `--tosa-to-linalg` pass has a bug so that we had to call it using
# `--pass-pipeline`, which cannot be combined with other named passes.
mlir-opt test/conv2d-linalg.mlir \
 --one-shot-bufferize="bufferize-function-boundaries" \
 --expand-realloc \
 --canonicalize \
 --ownership-based-buffer-deallocation \
 --buffer-deallocation-simplification \
 --bufferization-lower-deallocations \
 --canonicalize \
 --normalize-memrefs \
 --convert-linalg-to-affine-loops \
 --lower-affine \
 -o test/conv2d.mlir
```

NOTE: We do not support dynamically shaped memrefs at this time, so we should
manually add shapes to the memref type of the input argument (as well as other
dynamically shaped memref values inferred from the input argument). This can be
done by replacing ? with concrete numbers.

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


