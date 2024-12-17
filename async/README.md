# MLIR Async Dialect to Wasm Example

## compiling example.mlir to wasm

```
mlir-opt --async-func-to-async-runtime --async-to-async-runtime \
--convert-async-to-llvm \
--convert-to-llvm \
--reconcile-unrealized-casts \
--async-func-to-async-runtime --canonicalize \
example.mlir -o example-ll.mlir
```

```
mlir-translate example-ll.mlir --mlir-to-llvmir -o example.ll
```

```
opt --passes="coro-early,function(coro-elide),coro-split,coro-cleanup" example.ll -o example-lowered.bc
```

```
llc -O3 -filetype=obj -mtriple=wasm32-wasi example-lowered.bc -o example.o
```

## compiling async runtime to wasm

```
$WASI_SDK_PATH/bin/clang \
--target=wasm32-wasi \
--sysroot=$WASI_SDK_PATH/share/wasi-sysroot \
-O0 -c async_runtime.c -o async_runtime.o
```

## linking

```
$WASI_SDK_PATH/bin/wasm-ld \
  example.o async_runtime.o \
  --no-entry \
  --allow-undefined \
  --export=main \
  $WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasi/libc.a \
  -o example.wasm
```