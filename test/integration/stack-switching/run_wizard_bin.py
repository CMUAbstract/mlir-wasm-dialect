#!/usr/bin/env python3

import argparse
import functools
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


@functools.lru_cache(maxsize=1)
def resolve_java() -> Optional[Path]:
    candidates = []
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(Path(java_home) / "bin" / "java")

    java_from_path = shutil.which("java")
    if java_from_path:
        candidates.append(Path(java_from_path))

    # Common Homebrew install locations on macOS.
    candidates.append(Path("/opt/homebrew/opt/openjdk/bin/java"))
    candidates.append(Path("/usr/local/opt/openjdk/bin/java"))

    seen = set()
    for java in candidates:
        key = str(java)
        if key in seen:
            continue
        seen.add(key)
        if not java.is_file() or not os.access(java, os.X_OK):
            continue
        probe = subprocess.run(
            [str(java), "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return java
    return None


def has_working_java() -> bool:
    return resolve_java() is not None


def resolve_wizeng() -> Path:
    override = os.environ.get("WIZARD_WIZENG_BIN")
    if override:
        path = Path(override)
        if path.is_file() and os.access(path, os.X_OK):
            if path.name.endswith(".jvm") and not has_working_java():
                raise RuntimeError(
                    "WIZARD_WIZENG_BIN points to wizeng.jvm but no working Java runtime was found"
                )
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
        # Prefer JVM runner on macOS to avoid Rosetta/x86 compatibility issues.
        candidates.append(bindir / "wizeng.jvm")
        candidates.append(bindir / "wizeng.x86-64-darwin")

    # Generic fallbacks.
    candidates.append(bindir / "wizeng.jvm")
    candidates.append(bindir / "wizeng")

    for cand in candidates:
        if not cand.is_file() or not os.access(cand, os.X_OK):
            continue
        if cand.name.endswith(".jvm") and not has_working_java():
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

    cmd.extend([
        "--ext:all",
        "--expose=wizeng",
        f"--invoke={args.invoke}",
        "--print-result",
        args.input,
    ])

    if args.program_args:
        cmd.append("--")
        cmd.extend(args.program_args)

    run_env = os.environ.copy()
    if wizeng.name.endswith(".jvm"):
        java = resolve_java()
        if java is None:
            return fail("wizeng.jvm selected, but no working Java runtime was found")
        java_bin = str(java.parent)
        current_path = run_env.get("PATH", "")
        if not current_path:
            run_env["PATH"] = java_bin
        elif java_bin not in current_path.split(os.pathsep):
            run_env["PATH"] = java_bin + os.pathsep + current_path

    proc = subprocess.run(cmd, capture_output=True, text=True, env=run_env)

    if args.expect_i32 is not None:
        observed = extract_i32_result(proc.stdout)
        exit_matches = proc.returncode == args.expect_i32
        parse_matches = observed == args.expect_i32

        # Wizard may encode the main i32 result as process exit code.
        # Keep stdout parsing as the primary path, but allow the exit-code path.
        if proc.returncode != 0 and not exit_matches:
            if proc.stdout:
                print(proc.stdout, end="", file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            return fail(f"wizard execution failed with exit code {proc.returncode}")

        if not parse_matches and not exit_matches:
            if proc.stdout:
                print(proc.stdout, end="", file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            if observed is None:
                return fail("could not parse i32 result from wizard output")
            return fail(
                f"result mismatch: expected {args.expect_i32}, got {observed}"
            )
    else:
        if proc.returncode != 0:
            if proc.stdout:
                print(proc.stdout, end="", file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            return fail(f"wizard execution failed with exit code {proc.returncode}")

    if not args.quiet:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
