#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  test/integration/stack-switching/run_coro_program.sh \
    --mode {wami|llvm} \
    --input <program.mlir> \
    [--expect-i32 <n>] \
    [--invoke <name>] \
    [--quiet] \
    [--keep-tmp] \
    [--tmp-root <dir>]

Description:
  Builds and runs a coro MLIR program with Wizard using either:
  - wami: WAMI -> wasmstack -> wasm-emit
  - llvm: pure LLVM coroutine backend pipeline

Environment:
  - WIZARD_ENGINE_DIR or WIZARD_WIZENG_BIN should be set for execution.
  - LLVM_BIN_DIR (optional): directory containing mlir-opt, mlir-translate,
    opt, llc, wasm-ld for --mode llvm.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

resolve_tool() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return 0
  fi
  return 1
}

resolve_repo_tool() {
  local repo_root="$1"
  local name="$2"
  if [[ -x "$repo_root/build/bin/$name" ]]; then
    echo "$repo_root/build/bin/$name"
    return 0
  fi
  resolve_tool "$name" && return 0
  return 1
}

resolve_llvm_tool() {
  local name="$1"
  if [[ -n "${LLVM_BIN_DIR:-}" && -x "${LLVM_BIN_DIR}/$name" ]]; then
    echo "${LLVM_BIN_DIR}/$name"
    return 0
  fi
  if [[ -x "/Users/byeongjee/llvm-project/build/bin/$name" ]]; then
    echo "/Users/byeongjee/llvm-project/build/bin/$name"
    return 0
  fi
  resolve_tool "$name" && return 0
  return 1
}

MODE=""
INPUT=""
EXPECT_I32=""
INVOKE="main"
QUIET=0
KEEP_TMP=0
TMP_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --input)
      INPUT="${2:-}"
      shift 2
      ;;
    --expect-i32)
      EXPECT_I32="${2:-}"
      shift 2
      ;;
    --invoke)
      INVOKE="${2:-}"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    --keep-tmp)
      KEEP_TMP=1
      shift
      ;;
    --tmp-root)
      TMP_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ -n "$MODE" ]] || die "--mode is required"
[[ "$MODE" == "wami" || "$MODE" == "llvm" ]] || die "--mode must be 'wami' or 'llvm'"
[[ -n "$INPUT" ]] || die "--input is required"
[[ -f "$INPUT" ]] || die "input file not found: $INPUT"

if [[ -z "${WIZARD_ENGINE_DIR:-}" && -z "${WIZARD_WIZENG_BIN:-}" ]]; then
  die "set WIZARD_ENGINE_DIR (or WIZARD_WIZENG_BIN) before running"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

WASM_OPT="$(resolve_repo_tool "$REPO_ROOT" wasm-opt)" || die "could not resolve wasm-opt"
WASM_EMIT="$(resolve_repo_tool "$REPO_ROOT" wasm-emit)" || die "could not resolve wasm-emit"
RUNNER="${SCRIPT_DIR}/run_wizard_bin.py"
[[ -f "$RUNNER" ]] || die "runner not found: $RUNNER"

if [[ -n "$TMP_ROOT" ]]; then
  mkdir -p "$TMP_ROOT"
  WORKDIR="$(mktemp -d "$TMP_ROOT/coro-run.XXXXXX")"
else
  WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/coro-run.XXXXXX")"
fi

cleanup() {
  if [[ "$KEEP_TMP" -eq 0 ]]; then
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

OUT_WASM="${WORKDIR}/program.wasm"

if [[ "$MODE" == "wami" ]]; then
  "$WASM_OPT" "$INPUT" \
    --coro-verify-intrinsics \
    --coro-normalize \
    --wami-convert-all \
    --reconcile-unrealized-casts \
    --coro-to-wami \
    --convert-to-wasmstack \
    --verify-wasmstack \
    | "$WASM_EMIT" --mlir-to-wasm -o "$OUT_WASM"
else
  MLIR_OPT="$(resolve_llvm_tool mlir-opt)" || die "could not resolve mlir-opt"
  MLIR_TRANSLATE="$(resolve_llvm_tool mlir-translate)" || die "could not resolve mlir-translate"
  OPT_BIN="$(resolve_llvm_tool opt)" || die "could not resolve opt"
  LLC_BIN="$(resolve_llvm_tool llc)" || die "could not resolve llc"
  WASM_LD_BIN="$(resolve_llvm_tool wasm-ld)" || die "could not resolve wasm-ld"

  NORM_MLIR="${WORKDIR}/norm.mlir"
  PRELLVM_MLIR="${WORKDIR}/prellvm.mlir"
  CORO_LLVM_MLIR="${WORKDIR}/coro.llvm.mlir"
  LLVM_LL="${WORKDIR}/llvm.ll"
  LLVM_OPT_LL="${WORKDIR}/llvm.opt.ll"
  LLVM_OBJ="${WORKDIR}/llvm.o"

  "$WASM_OPT" "$INPUT" \
    --coro-verify-intrinsics \
    --coro-normalize \
    -o "$NORM_MLIR"

  "$MLIR_OPT" "$NORM_MLIR" \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-arith-to-llvm="index-bitwidth=32" \
    --convert-func-to-llvm="index-bitwidth=32" \
    --memref-expand \
    --expand-strided-metadata \
    --finalize-memref-to-llvm="index-bitwidth=32" \
    --convert-cf-to-llvm="index-bitwidth=32" \
    --convert-to-llvm \
    --reconcile-unrealized-casts \
    -o "$PRELLVM_MLIR"

  "$WASM_OPT" "$PRELLVM_MLIR" --coro-to-llvm -o "$CORO_LLVM_MLIR"
  "$MLIR_TRANSLATE" "$CORO_LLVM_MLIR" --mlir-to-llvmir -o "$LLVM_LL"
  "$OPT_BIN" -passes='coro-early,coro-split,coro-elide,coro-cleanup' "$LLVM_LL" -o "$LLVM_OPT_LL"
  "$LLC_BIN" -filetype=obj -mtriple=wasm32-wasi "$LLVM_OPT_LL" -o "$LLVM_OBJ"
  "$WASM_LD_BIN" --no-entry --allow-undefined --export=main --export-memory -o "$OUT_WASM" "$LLVM_OBJ"
fi

RUN_CMD=(python3 "$RUNNER" --input "$OUT_WASM" --invoke "$INVOKE")
if [[ -n "$EXPECT_I32" ]]; then
  RUN_CMD+=(--expect-i32 "$EXPECT_I32")
fi
if [[ "$QUIET" -eq 1 ]]; then
  RUN_CMD+=(--quiet)
fi

"${RUN_CMD[@]}"

if [[ "$KEEP_TMP" -eq 1 ]]; then
  echo "Kept artifacts: $WORKDIR"
fi
