# Integration Benchmarks (Opt-in)

These tests validate end-to-end execution:

1. High-level MLIR (`arith`, `memref`, `scf`, `func`)
2. `wami-convert-all`
3. `reconcile-unrealized-casts`
4. `convert-to-wasmstack`
5. `wasm-emit --mlir-to-wasm`
6. Execute generated wasm with `wasmtime-executor`
7. Assert exact `main() -> i32` result

## Run

```bash
RUN_WASMTIME_BENCH=1 llvm-lit test/integration/benchmarks
```

The suite is intentionally opt-in and non-gating by default.
