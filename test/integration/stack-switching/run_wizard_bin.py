#!/usr/bin/env python3

import argparse
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def resolve_wizeng() -> Path:
    override = os.environ.get("WIZARD_WIZENG_BIN")
    if override:
        path = Path(override)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise RuntimeError(f"WIZARD_WIZENG_BIN is not executable: {path}")

    wizard_dir = os.environ.get("WIZARD_ENGINE_DIR")
    if not wizard_dir:
        raise RuntimeError("WIZARD_ENGINE_DIR is not set")

    root = Path(wizard_dir)
    bindir = root / "bin"

    sys_name = platform.system().lower()
    machine = platform.machine().lower()

    candidates = []
    if sys_name == "linux":
        if machine in ("x86_64", "amd64"):
            candidates.append(bindir / "wizeng.x86-64-linux")
        elif machine.startswith("i") and machine.endswith("86"):
            candidates.append(bindir / "wizeng.x86-linux")
    elif sys_name == "darwin":
        candidates.append(bindir / "wizeng.x86-64-darwin")
        candidates.append(bindir / "wizeng.jvm")

    # Generic fallbacks.
    candidates.append(bindir / "wizeng.jvm")
    candidates.append(bindir / "wizeng")

    for cand in candidates:
        if not cand.is_file() or not os.access(cand, os.X_OK):
            continue
        if cand.name.endswith(".jvm") and shutil.which("java") is None:
            continue
        return cand

    raise RuntimeError(
        "could not find an executable wizeng binary under "
        f"{bindir}; set WIZARD_WIZENG_BIN to override"
    )


def extract_i32_result(stdout: str):
    # With --print-result, wizeng prints one line with the main result values.
    values = []
    for line in stdout.splitlines():
        m = re.fullmatch(r"\s*(-?\d+)\s*", line)
        if m:
            values.append(int(m.group(1)))
    if not values:
        return None
    return values[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a wasm module with wizard engine and optionally check i32 result"
    )
    parser.add_argument("--input", required=True, help="input wasm file")
    parser.add_argument("--expect-i32", type=int, help="expected i32 result")
    parser.add_argument("--invoke", default="main", help="entrypoint function name")
    parser.add_argument("--quiet", action="store_true", help="suppress non-error output")
    parser.add_argument("program_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    try:
        wizeng = resolve_wizeng()
    except RuntimeError as err:
        return fail(str(err))

    cmd = [str(wizeng)]
    extra_opts = os.environ.get("WIZARD_WIZENG_OPTS", "")
    if extra_opts:
        cmd.extend(shlex.split(extra_opts))

    cmd.extend(["--ext:all", f"--invoke={args.invoke}", "--print-result", args.input])

    if args.program_args:
        cmd.append("--")
        cmd.extend(args.program_args)

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        return fail(f"wizard execution failed with exit code {proc.returncode}")

    if args.expect_i32 is not None:
        observed = extract_i32_result(proc.stdout)
        if observed is None:
            if proc.stdout:
                print(proc.stdout, end="", file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            return fail("could not parse i32 result from wizard output")
        if observed != args.expect_i32:
            if proc.stdout:
                print(proc.stdout, end="", file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            return fail(
                f"result mismatch: expected {args.expect_i32}, got {observed}"
            )

    if not args.quiet:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
