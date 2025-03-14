# How to compile toy programs

Both `toy.mlir` and `toy-llvm.mlir` are programs with the same functionality: a
simple yield-style generator.

## toy.mlir

We can compile this using our Wasm dialect and dialect conversion passes.

```sh
$BUILD_DIR/bin/wasm-opt --convert-dcont-to-ssawasm test/DCont/toy.mlir \
--convert-arith-to-ssawasm --convert-func-to-ssawasm --convert-memref-to-ssawasm \
--convert-scf-to-ssawasm --reconcile-unrealized-casts \
--convert-ssawasm-global-to-wasm --introduce-locals --convert-ssawasm-to-wasm \
-o toy-wasm.mlir

$BUILD_DIR/bin/wasm-translate --mlir-to-wat toy-wasm.mlir -o toy.wat
```

The output `toy.wat` can be run on [wasmfxtime](https://github.com/wasmfx/wasmfxtime),
which is the reference runtime implementation for the stack switching proposal.

## toy-llvm.mlir

This uses LLVM coroutine intrinsics and passes to implement an yield-style generator.

```sh
mlir-translate toy-llvm.mlir --mlir-to-llvmir -o toy-llvm.ll

opt --passes="coro-early,function(coro-elide),coro-split,coro-cleanup" \
toy-llvm.ll -o toy-llvm-lowered.bc

opt -O3 toy-llvm-lowered.bc -o toy-llvm-lowered-opt.bc

opt --passes="coro-early,function(coro-elide),coro-split,coro-cleanup" \
toy-llvm-lowered-opt.bc -o toy-llvm-lowered-opt-processed.bc


llc -O3 toy-llvm-lowered.bc -filetype=obj -mtriple=wasm32-wasi -o toy-llvm-lowered.o

$WASI_SDK_PATH/bin/wasm-ld --no-entry \
   --export=main --allow-undefined \
    -L $WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasi -lc \
    -o toy-llvm.wasm toy-llvm-lowered.o


mlir-translate toy-scheduler-llvm.mlir --mlir-to-llvmir -o toy-scheduler-llvm.ll

opt --passes="coro-early,function(coro-elide),coro-split,coro-cleanup" \
toy-scheduler-llvm.ll -o toy-scheduler-llvm-lowered.bc

llc -O3 toy-scheduler-llvm-lowered.bc -filetype=obj -mtriple=wasm32-wasi -o toy-scheduler-llvm-lowered.o

$WASI_SDK_PATH/bin/wasm-ld --no-entry \
   --export=main --allow-undefined \
    -L $WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasi -lc \
    -o toy-scheduler-llvm.wasm toy-scheduler-llvm-lowered.o
```

The output `toy-llvm.wasm` can be run on any Wasm runtime.