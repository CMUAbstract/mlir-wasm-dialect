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
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_ASSERTIONS=ON
cmake --build . --target check-mlir
```

## Build

```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-wasm
```

## Run
```sh
export PATH="bin:$PATH"
wasm-opt --convert-to-wasm --reconcile-unrealized-casts --wasm-finalize input.mlir
```
