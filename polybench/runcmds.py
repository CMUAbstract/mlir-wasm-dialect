#!/usr/bin/env python3

import sys, signal, socket, time, threading, subprocess, multiprocessing, os, json
from datetime import datetime

NCPU = multiprocessing.cpu_count()

_currentChildren = []


def _signalHandler(signum, frame):
    global _currentChildren
    sys.stderr.write("[ERR] Interrupted.\n")
    if _currentChildren:
        for child in _currentChildren:
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            except Exception as e:
                sys.stderr.write(
                    "[WARN] Error while trying to kill process {}: {}\n".format(
                        child.pid, str(e)
                    )
                )
    sys.exit(1)


signal.signal(signal.SIGINT, _signalHandler)


def _killer():
    global _currentChildren
    if _currentChildren:
        for child in _currentChildren:
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            except Exception as e:
                sys.stderr.write(
                    "[WARN] Error while trying to kill process {}: {}\n".format(
                        child.pid, str(e)
                    )
                )


def runcmds(rows, timeout=600.0, silent=False):
    global _currentChildren
    numCmds = len(rows)
    cmdNum = 0

    for row in rows:
        row = row.copy()
        cmdNum += 1
        row["host"] = socket.gethostname()
        row["timestamp"] = datetime.now().strftime("%y-%m-%d %H:%M:%S.%f")

        if not silent:
            sys.stderr.write("[{}/{}] {}\n".format(cmdNum, numCmds, row["cmd"]))

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
        _currentChildren.append(subproc)

        timer = threading.Timer(timeout, _killer)
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
        _currentChildren = []

        yield row


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timeout", type=float, default=300.0, dest="timeout")
    parser.add_argument("-s", "--silent", action="store_true", dest="silent")
    parser.add_argument(
        "-o", "--output", type=argparse.FileType("a"), default=sys.stdout, dest="output"
    )
    args = parser.parse_args()

    rows = [json.loads(x) for x in sys.stdin]
    for result in runcmds(rows, timeout=args.timeout, silent=args.silent):
        s = "{}\n".format(json.dumps(result))
        args.output.write(s)
        args.output.flush()

        if not args.silent:
            sys.stderr.write(result["stdout"] + "\n")
            sys.stderr.write(result["stderr"] + "\n")
