# MLIR WebAssembly Dialect

## Setup

Build LLVM and MLIR in `$BUILD_DIR` and install them to `$PREFIX`.
For example, I use the following:
```sh
export BUILD_DIR=/Users/byeongje/wasm/llvm-project/build
export PREFIX=/Users/byeongje/wasm/llvm-project/build
```

LLVM can be built by running the following command in LLVM root directory.
```sh
mkdir build && cd build
cmake -G Ninja ../llvm \
-DLLVM_ENABLE_PROJECTS=mlir \
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly \
-DCMAKE_BUILD_TYPE=Debug \
-DLLVM_ENABLE_ASSERTIONS=ON
cmake --build . --target check-mlir
```

## Build

```sh
mkdir build && cd build
cmake -G Ninja ..  -DMLIR_DIR=$PREFIX/lib/cmake/mlir  -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit -DCMAKE_BUILD_TYPE=Debug 
cmake --build . --target check-wasm
```

## Run
```sh
export PATH="build/bin:$PATH"
wasm-opt --convert-to-wasm --reconcile-unrealized-casts --wasm-finalize input.mlir
```

## Target
For the MVP, we aim to support lowering `test/conv2d.mlir` to the wasm dialect.
The original MLIR file, `test/conv2d-tosa.mlir`, is a simplified version of an MLIR file produced by TFLite.
We apply the following pipelines to generate `test/conv2d.mlir`:
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
mlir-opt conv2d-linalg.mlir \
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
 -o conv2d.mlir
```

## Baseline
The baseline for comparison (`conv2d.wat`) is produced as follows:
```sh
mlir-opt conv2d.mlir --convert-scf-to-cf --lower-affine --convert-arith-to-llvm="index-bitwidth=32" --convert-func-to-llvm="index-bitwidth=32" --memref-expand --expand-strided-metadata --finalize-memref-to-llvm="index-bitwidth=32" --convert-to-llvm --reconcile-unrealized-casts -o conv2d-llvm.mlir
```

```sh
mlir-translate conv2d-llvm.mlir --mlir-to-llvmir -o conv2d.ll
```

```sh
llc -O0 -filetype=obj -mtriple=wasm32-wasi conv2d.ll -o conv2d.o
```

```sh
wasm2wat conv2d.o -o conv2d.wat
```