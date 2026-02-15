# -*- Python -*-

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "WASM"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.wasm_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%wasm_src_root", config.wasm_src_root))

llvm_config.with_system_environment([
    "HOME",
    "INCLUDE",
    "LIB",
    "TMP",
    "TEMP",
    "RUN_WIZARD_STACK_SWITCHING",
    "WIZARD_ENGINE_DIR",
    "WIZARD_WIZENG_BIN",
    "WIZARD_WIZENG_OPTS",
])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.wasm_obj_root, "test")
config.wasm_tools_dir = os.path.join(config.wasm_obj_root, "bin")
config.wasm_libs_dir = os.path.join(config.wasm_obj_root, "lib")

config.substitutions.append(("%wasm_libs", config.wasm_libs_dir))

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.wasm_tools_dir, config.llvm_tools_dir]
tools = [
    "mlir-opt",
    "wasm-opt",
    "wasm-translate",
    "wasm-emit",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Optional wabt tools used by integration tests.
wabt_tools = [
    ToolSubst("wasm-validate", command=FindTool("wasm-validate"),
              unresolved="ignore"),
    ToolSubst("wasm-objdump", command=FindTool("wasm-objdump"),
              unresolved="ignore"),
]
llvm_config.add_tool_substitutions(
    wabt_tools, tool_dirs + [config.environment.get("PATH", "")]
)

if all(tool.was_resolved for tool in wabt_tools):
    config.available_features.add("wabt")

# Optional wasm-ld used by relocatable/linking tests.
wasm_ld_tool = ToolSubst(
    "wasm-ld", command=FindTool("wasm-ld"), unresolved="ignore"
)
llvm_config.add_tool_substitutions(
    [wasm_ld_tool], tool_dirs + [config.environment.get("PATH", "")]
)
if wasm_ld_tool.was_resolved:
    config.available_features.add("wasm_ld")

# Optional LLVM-backend tools used by pure LLVM -> Wasm integration tests.
llvm_backend_tools = [
    ToolSubst("mlir-translate", command=FindTool("mlir-translate"),
              unresolved="ignore"),
    ToolSubst("llc", command=FindTool("llc"), unresolved="ignore"),
]
llvm_config.add_tool_substitutions(
    llvm_backend_tools, tool_dirs + [config.environment.get("PATH", "")]
)
if (all(tool.was_resolved for tool in llvm_backend_tools) and
        wasm_ld_tool.was_resolved):
    config.available_features.add("llvm_wasm_backend")

# Optional wasmtime execution integration tests (opt-in).
# Enabled only when cargo is available and RUN_WASMTIME_BENCH=1.
run_wasmtime_bench = os.environ.get("RUN_WASMTIME_BENCH", "") == "1"
cargo = shutil.which("cargo")
if run_wasmtime_bench and cargo:
    config.available_features.add("wasmtime_exec")
    run_wasm_manifest = os.path.join(
        config.wasm_src_root, "wasmtime-executor", "Cargo.toml"
    )
    run_wasm_cmd = (
        f"{cargo} run --quiet --manifest-path {run_wasm_manifest} "
        "--bin run_wasm_bin --"
    )
    config.substitutions.append(("%run_wasm_bin", run_wasm_cmd))

# Optional wizard-engine execution integration tests (opt-in).
# Enabled only when WIZARD_ENGINE_DIR is set and RUN_WIZARD_STACK_SWITCHING=1.
run_wizard_stack = os.environ.get("RUN_WIZARD_STACK_SWITCHING", "") == "1"
wizard_engine_dir = os.environ.get("WIZARD_ENGINE_DIR", "")
wizard_runner = os.path.join(
    config.wasm_src_root, "test", "integration", "stack-switching",
    "run_wizard_bin.py"
)
if (run_wizard_stack and wizard_engine_dir and
        os.path.isdir(wizard_engine_dir) and
        os.path.isfile(wizard_runner)):
    config.available_features.add("wizard_exec")
    config.substitutions.append(
        ("%run_wizard_bin", f"{sys.executable} {wizard_runner}")
    )

llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.mlir_obj_dir, "python_packages", "wasm"),
    ],
    append_path=True,
)
