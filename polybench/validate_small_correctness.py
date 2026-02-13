#!/usr/bin/env python3
"""Differential correctness validation for polybench/small.

For each benchmark:
1) Compile with --compiler=wami
2) Compile with --compiler=llvm
3) Run both artifacts with wasmtime-executor in print hash mode
4) Compare return value + print stream hash + print count
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
SMALL_DIR = ROOT_DIR / "polybench" / "small"
COMPILE_SCRIPT = ROOT_DIR / "compile.sh"
EXECUTOR_DIR = ROOT_DIR / "wasmtime-executor"
EXECUTOR_BIN = EXECUTOR_DIR / "target" / "release" / "run_wasm_bin"
DEFAULT_HASH_SEED = 14695981039346656037


@dataclass
class CommandResult:
    stdout: str
    stderr: str


class CommandError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, stdout: str, stderr: str):
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        cmd_text = " ".join(self.cmd)
        return (
            f"command failed (exit={self.returncode}): {cmd_text}\n"
            f"stdout:\n{self.stdout}\n"
            f"stderr:\n{self.stderr}"
        )


def run_cmd(cmd: Sequence[str], cwd: Path) -> CommandResult:
    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CommandError(cmd, completed.returncode, completed.stdout, completed.stderr)
    return CommandResult(stdout=completed.stdout, stderr=completed.stderr)


def build_executor() -> None:
    run_cmd(["cargo", "+nightly", "build", "--quiet", "--release"], EXECUTOR_DIR)
    if not EXECUTOR_BIN.exists():
        raise RuntimeError(f"expected executor binary not found: {EXECUTOR_BIN}")


def compile_wasm(
    mlir_path: Path,
    compiler: str,
    out_dir: Path,
    binaryen_opt_flags: str,
    llvm_opt_flags: str,
) -> Path:
    output_base = out_dir / f"{mlir_path.stem}-{compiler}"
    cmd: List[str] = [
        str(COMPILE_SCRIPT),
        "-i",
        str(mlir_path),
        "-o",
        str(output_base),
        f"--compiler={compiler}",
    ]
    if binaryen_opt_flags:
        cmd.append(f"--binaryen-opt-flags={binaryen_opt_flags}")
    if compiler == "llvm" and llvm_opt_flags:
        cmd.append(f"--llvm-opt-flags={llvm_opt_flags}")
    run_cmd(cmd, ROOT_DIR)

    wasm_path = output_base.with_suffix(".wasm")
    if not wasm_path.exists():
        raise RuntimeError(f"expected wasm output not found: {wasm_path}")
    return wasm_path


def parse_json_report(stdout: str) -> Dict[str, object]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"failed to find JSON report in stdout:\n{stdout}")


def run_wasm(wasm_path: Path, hash_seed: int) -> Dict[str, object]:
    result = run_cmd(
        [
            str(EXECUTOR_BIN),
            "--input",
            str(wasm_path),
            "--json",
            "--quiet",
            "--print-mode",
            "hash",
            "--print-hash-seed",
            str(hash_seed),
        ],
        EXECUTOR_DIR,
    )
    return parse_json_report(result.stdout)


def discover_small_benchmarks(name_filter: str) -> List[Path]:
    matcher = re.compile(name_filter)
    matches = [
        path
        for path in sorted(SMALL_DIR.glob("*.mlir"))
        if matcher.search(path.stem) is not None
    ]
    return matches


def compare_reports(lhs: Dict[str, object], rhs: Dict[str, object]) -> Dict[str, tuple]:
    mismatches = {}
    for key in ("actual", "print_count", "print_hash"):
        if lhs.get(key) != rhs.get(key):
            mismatches[key] = (lhs.get(key), rhs.get(key))
    return mismatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate polybench/small correctness by comparing WAMI and LLVM outputs."
    )
    parser.add_argument(
        "--filter",
        default=".*",
        help="Regex over benchmark name (default: all small benchmarks).",
    )
    parser.add_argument(
        "--print-hash-seed",
        type=int,
        default=DEFAULT_HASH_SEED,
        help=f"Hash seed for print stream hashing (default: {DEFAULT_HASH_SEED}).",
    )
    parser.add_argument(
        "--binaryen-opt-flags",
        default="",
        help="Optional Binaryen optimization flags forwarded to compile.sh.",
    )
    parser.add_argument(
        "--llvm-opt-flags",
        default="",
        help="Optional LLVM optimization flags forwarded to compile.sh for --compiler=llvm.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep per-benchmark temporary directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not COMPILE_SCRIPT.exists():
        print(f"error: compile script not found: {COMPILE_SCRIPT}", file=sys.stderr)
        return 2
    if not SMALL_DIR.is_dir():
        print(f"error: small benchmark directory not found: {SMALL_DIR}", file=sys.stderr)
        return 2
    if "WASI_SDK_PATH" not in os.environ or not os.environ["WASI_SDK_PATH"]:
        print("error: WASI_SDK_PATH must be set for llvm compilation.", file=sys.stderr)
        return 2

    benchmarks = discover_small_benchmarks(args.filter)
    if not benchmarks:
        print(f"no benchmarks matched filter: {args.filter}")
        return 1

    build_executor()

    failures = 0
    preserved_dirs: List[Path] = []
    print(f"running {len(benchmarks)} small benchmarks")

    for benchmark in benchmarks:
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"tmp_validate_{benchmark.stem}_"))
        keep_dir = args.keep_temp
        try:
            wami_wasm = compile_wasm(
                benchmark,
                "wami",
                tmp_dir,
                args.binaryen_opt_flags,
                args.llvm_opt_flags,
            )
            llvm_wasm = compile_wasm(
                benchmark,
                "llvm",
                tmp_dir,
                args.binaryen_opt_flags,
                args.llvm_opt_flags,
            )
            wami_report = run_wasm(wami_wasm, args.print_hash_seed)
            llvm_report = run_wasm(llvm_wasm, args.print_hash_seed)
            mismatches = compare_reports(wami_report, llvm_report)

            if mismatches:
                failures += 1
                keep_dir = True
                print(f"[FAIL] {benchmark.stem}")
                for key, (wami_value, llvm_value) in mismatches.items():
                    print(f"  {key}: wami={wami_value} llvm={llvm_value}")
                print(f"  temp_dir: {tmp_dir}")
            else:
                print(
                    f"[PASS] {benchmark.stem} "
                    f"actual={wami_report['actual']} "
                    f"count={wami_report['print_count']} "
                    f"hash={wami_report['print_hash']}"
                )
        except (CommandError, RuntimeError, json.JSONDecodeError) as exc:
            failures += 1
            keep_dir = True
            print(f"[ERROR] {benchmark.stem}: {exc}")
            print(f"  temp_dir: {tmp_dir}")

        if keep_dir:
            preserved_dirs.append(tmp_dir)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    passed = len(benchmarks) - failures
    print(f"summary: passed={passed} failed={failures} total={len(benchmarks)}")
    if preserved_dirs:
        print("preserved temp dirs:")
        for path in preserved_dirs:
            print(f"  {path}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
