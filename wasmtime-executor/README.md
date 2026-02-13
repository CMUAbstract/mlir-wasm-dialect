# Wasmtime Executor

Deterministic WebAssembly runner used by integration tests and benchmarks.

## Usage

```bash
cargo run -- --input <file.wasm> --expect-i32 <value>
```

### Options

- `--input <path>`: wasm module file path.
- `--entry <symbol>`: entry function name (default: `main`).
- `--expect-i32 <value>`: expected return value for `() -> i32` entry.
- `--iterations <N>`: measured iterations (default: `1`).
- `--warmup <N>`: warmup iterations (default: `0`).
- `--quiet`: suppress non-essential text output.
- `--json`: emit one-line JSON report.
- `--print-mode <normal|hash>`: print values normally (`normal`) or suppress
  prints while hashing/counting the `print_i32` stream (`hash`).
- `--print-hash-seed <u64>`: initial seed for print stream hash.

## Exit Codes

- `0`: success.
- `2`: module load/instantiation error.
- `3`: entry function not found.
- `4`: invalid args or signature mismatch.
- `5`: runtime trap.
- `6`: expected result mismatch.

## Host Imports

The runner provides host functions from `env`:

- `malloc(i32) -> i32`
- `free(i32) -> ()`
- `print_i32(i32) -> ()`
- `toggle_gpio() -> ()`

## Output Fields

Both text and JSON reports include:

- `print_count`: number of `print_i32` calls observed in the run.
- `print_hash`: deterministic 64-bit hash of the `print_i32` value stream.
