#!/usr/bin/env python3

import sys
import signal
import socket
import time
import threading
import subprocess
import multiprocessing
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

NCPU = multiprocessing.cpu_count()

_childrenLock = threading.Lock()
_stderrLock = threading.Lock()
_currentChildren = set()


def _killAllChildren():
    with _childrenLock:
        for child in _currentChildren:
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            except Exception as e:
                sys.stderr.write(
                    "[WARN] Error while trying to kill process {}: {}\n".format(
                        child.pid, str(e)
                    )
                )


def _signalHandler(signum, frame):
    sys.stderr.write("[ERR] Interrupted.\n")
    _killAllChildren()
    sys.exit(1)


signal.signal(signal.SIGINT, _signalHandler)


def _run_single(row, timeout, silent, idx, total):
    """Run a single command row. Returns the enriched row dict."""
    row = row.copy()
    row["host"] = socket.gethostname()
    row["timestamp"] = datetime.now().strftime("%y-%m-%d %H:%M:%S.%f")

    if not silent:
        with _stderrLock:
            sys.stderr.write("[{}/{}] {}\n".format(idx, total, row["cmd"]))

    # Launch the command
    if "cwd" in row:
        subproc = subprocess.Popen(
            row["cmd"],
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            cwd=row["cwd"],
        )
    else:
        subproc = subprocess.Popen(
            row["cmd"],
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

    with _childrenLock:
        _currentChildren.add(subproc)

    def _timeout_killer(proc):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

    timer = threading.Timer(timeout, _timeout_killer, args=[subproc])
    ts = time.time()

    row["stdout"] = ""
    row["stderr"] = ""

    try:
        timer.start()
        row["stdout"], row["stderr"] = subproc.communicate()
    finally:
        timer.cancel()
        row["elapsed"] = time.time() - ts

    row["returncode"] = subproc.returncode

    with _childrenLock:
        _currentChildren.discard(subproc)

    return row


def runcmds(rows, timeout=600.0, silent=False):
    numCmds = len(rows)

    for idx, row in enumerate(rows, 1):
        yield _run_single(row, timeout, silent, idx, numCmds)


def runcmds_parallel(rows, timeout=600.0, silent=False, jobs=2):
    numCmds = len(rows)
    futures = {}

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        for idx, row in enumerate(rows, 1):
            future = executor.submit(_run_single, row, timeout, silent, idx, numCmds)
            futures[future] = idx

        for future in as_completed(futures):
            yield future.result()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timeout", type=float, default=600.0, dest="timeout")
    parser.add_argument("-s", "--silent", action="store_true", dest="silent")
    parser.add_argument(
        "-o", "--output", type=argparse.FileType("a"), default=sys.stdout, dest="output"
    )
    parser.add_argument("-j", "--jobs", type=int, default=1, dest="jobs")
    args = parser.parse_args()

    rows = [json.loads(x) for x in sys.stdin]

    # Validate device compatibility when using parallel execution
    if args.jobs > 1:
        for row in rows:
            device = row.get("device", "")
            if device not in ("local_wasmtime", "local_node"):
                tag = row.get("tag", "<unknown>")
                sys.stderr.write(
                    "Error: --jobs > 1 only supports device=local_wasmtime. "
                    "Found device='{}' in row for '{}'.\n".format(device, tag)
                )
                sys.exit(1)

    if args.jobs > 1:
        results = runcmds_parallel(
            rows, timeout=args.timeout, silent=args.silent, jobs=args.jobs
        )
    else:
        results = runcmds(rows, timeout=args.timeout, silent=args.silent)

    for result in results:
        s = "{}\n".format(json.dumps(result))
        args.output.write(s)
        args.output.flush()

        if not args.silent:
            sys.stderr.write(result["stdout"] + "\n")
            sys.stderr.write(result["stderr"] + "\n")
