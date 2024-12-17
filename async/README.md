# MLIR Async Dialect to Wasm Example

## compiling example.mlir to wasm

```sh
mlir-opt --async-func-to-async-runtime --async-to-async-runtime \
--convert-to-llvm example.mlir -o example-manualedit.mlir
```

### manual modification: remove these three lines from example-manualedit.mlir
```
# line 2-4
  llvm.func @abort()
  llvm.func @puts(!llvm.ptr)
  llvm.mlir.global private constant @assert_msg(dense<[65, 119, 97, 105, 116, 101, 100, 32, 97, 115, 121, 110, 99, 32, 111, 112, 101, 114, 97, 110, 100, 32, 105, 115, 32, 105, 110, 32, 101, 114, 114, 111, 114, 32, 115, 116, 97, 116, 101, 0]> : tensor<40xi8>) {addr_space = 0 : i32} : !llvm.array<40 x i8>
# line 64-68
    %4 = llvm.mlir.addressof @assert_msg : !llvm.ptr
    %5 = llvm.getelementptr %4[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<40 x i8>
    llvm.call @puts(%5) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
```

```
mlir-opt --convert-async-to-llvm \
--reconcile-unrealized-casts \
--async-func-to-async-runtime --canonicalize \
example-manualedit.mlir -o example-ll.mlir
```

manual modification required

after modification

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