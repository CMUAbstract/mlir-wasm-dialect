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

To convert an MLIR file (input.mlir) into the Wasm dialect, use the following
command:
```sh
build/bin/wasm-opt --convert-to-wasm --reconcile-unrealized-casts --wasm-finalize input.mlir -o output.mlir
```

The resulting MLIR file (`output.mlir`) can be translated into a textual
WebAssembly format (`.wat`) using:
```sh
build/bin/wasm-translate output.mlir --mlir-to-wat -o output.wat
```

Note: The generated `.wat` file may not be well-formatted. While there is no
standard formatter for `.wat` files, you can improve its readability by converting
the .wat file to a `.wasm` binary and then back to .wat using wasm2wat. Be aware
that this process might strip off symbol names.
```sh
wat2wasm --relocatable output.wat -o output.wasm
wasm2wat output.wasm -o output-formatted.wat
```

To further optimize the WebAssembly output (e.g., to reduce the number of local
variables), you can use the `wasm-opt` tool from Binaryen (note: this is different
from the wasm-opt binary we compiled):
```sh
wasm-opt output.wat -O4 output-optimized.wasm
```

### Running wasm file

We first need to convert the `wat` file into a relocatable object and run
`wasm-ld` to link it with stdlib.
```sh
wat2wasm --relocatable ./test/output.wat -o ./test/output.o
$WASI_SDK_PATH/bin/wasm-ld --no-entry \
--export-memory --export=main --export=malloc --export=free \
-L $WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasi -lc \
-o ./test/output-linked.wasm ./test/output.o
```

The generated output-linked.wasm can be executed using any wasm runtime. An
example runtime can be found in the run-wasm directory.



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

## Baseline
The baseline for comparison (`conv2d.wat`) is produced as follows:
```sh
mlir-opt ./test/conv2d.mlir --convert-scf-to-cf --lower-affine --convert-arith-to-llvm="index-bitwidth=32" --convert-func-to-llvm="index-bitwidth=32" --memref-expand --expand-strided-metadata --finalize-memref-to-llvm="index-bitwidth=32" --convert-to-llvm --reconcile-unrealized-casts -o ./test/conv2d-llvm.mlir
mlir-translate ./test/conv2d-llvm.mlir --mlir-to-llvmir -o ./test/conv2d.ll
llc -O0 -filetype=obj -mtriple=wasm32-wasi ./test/conv2d.ll -o ./test/conv2d.o
wasm2wat ./test/conv2d.o -o ./test/conv2d.wat
```

We need to link the wasm file with stdlib
```sh
$WASI_SDK_PATH/bin/wasm-ld --no-entry \
--export-memory --export=_mlir_ciface_main --export=malloc --export=free \
-L $WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasi -lc \
-o ./test/conv2d-linked.wasm ./test/conv2d.o
wasm2wat ./test/conv2d-linked.wasm -o ./test/conv2d-linked.wat
```
