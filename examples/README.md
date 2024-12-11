# Async example compilation pipeline to Native Code

```
mlir-opt --async-func-to-async-runtime --async-to-async-runtime --convert-async-to-llvm --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --convert-arith-to-llvm  --finalize-memref-to-llvm --convert-to-llvm --reconcile-unrealized-casts --async-func-to-async-runtime --canonicalize async.mlir -o async-ll.mlir

mlir-translate async-ll.mlir --mlir-to-llvmir -o async.ll

opt --passes="coro-early,function(coro-elide),coro-split,coro-cleanup" \
async.ll -o async-lowered.bc

llc async-lowered.bc -filetype=obj -o async-lowered.o

clang async-lowered.o -o exec -v -isysroot $(xcrun --show-sdk-path) -L/Users/byeongje/wasm/llvm-project/build/lib/ -lmlir_async_runtime
```
