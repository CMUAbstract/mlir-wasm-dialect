#!/usr/bin/env python3
import json

FILENAME = {
    "atax": "atax_256.mlir",
    "bicg": "bicg_256.mlir",
    "doitgen": "doitgen_64.mlir",
    "gemm": "gemm_128.mlir",
    "gemver": "gemver_256.mlir",
    "gesummv": "gesummv_256.mlir",
    "mvt": "mvt_256.mlir",
    "symm": "symm_128.mlir",
    "syr2k": "syr2k_128.mlir",
    "three_mm": "three_mm_64.mlir",
    "trmm": "trmm_128.mlir",
    "two_mm": "two_mm_128.mlir",
}


def gen_testcase(tag: str, compiler: str) -> str:
    return f"{tag.upper()}_{compiler.upper()}"


def cmd(
    tag: str,
    compiler: str,
    use_aot: bool,
    *,
    llvm_opt_level: int,
    binaryen_opt_level: int,
    aot_opt_level: int = -1,
) -> str:
    llvm_opt_flags = f"-O{llvm_opt_level}"
    binaryen_opt_flags = f"-O{binaryen_opt_level}"
    aot_str = (
        f"-- --opt-level={aot_opt_level} --target=thumbv7em --target-abi=eabihf --cpu=cortex-m4"
        if use_aot
        else ""
    )
    file_name = FILENAME[tag]
    testcase = gen_testcase(tag, compiler)
    cmd_template = f"cd .. && ./run-mcu.sh polybench/{file_name} --compiler={compiler} --testcase={testcase} --llvm-opt-flags={llvm_opt_flags} --binaryen-opt-flags={binaryen_opt_flags} --use-aot={use_aot} {aot_str}"
    return cmd_template


def make_row(
    tag: str,
    compiler: str,
    use_aot: bool,
    *,
    llvm_opt_level: int,
    binaryen_opt_level: int,
    aot_opt_level=-1,
) -> dict:
    row = {
        "tag": tag,
        "cmd": cmd(
            tag,
            compiler,
            use_aot,
            llvm_opt_level=llvm_opt_level,
            binaryen_opt_level=binaryen_opt_level,
            aot_opt_level=aot_opt_level,
        ),
        "compiler": compiler,
        "use_aot": use_aot,
        "llvm_opt_level": llvm_opt_level,
        "binaryen_opt_level": binaryen_opt_level,
    }
    if use_aot:
        row["aot_opt_level"] = aot_opt_level
    return row


uniques = set()


def filter_unique(row):
    if row is None:
        return None
    if row["cmd"] in uniques:
        return None
    uniques.add(row["cmd"])
    return row


if __name__ == "__main__":
    tags = [
        "atax",
        "bicg",
        "doitgen",
        "gemm",
        "gemver",
        "gesummv",
        "mvt",
        # "symm",
        "syr2k",
        "three_mm",
        "trmm",
        "two_mm",
    ]

    tests = [
        filter_unique(
            make_row(
                tag,
                compiler,
                use_aot,
                llvm_opt_level=llvm_opt_level,
                binaryen_opt_level=binaryen_opt_level,
                aot_opt_level=aot_opt_level,
            )
        )
        for tag in tags
        for compiler in ["mlir", "llvm"]
        for use_aot in [True, False]
        for llvm_opt_level in [0, 1, 2, 3]
        for binaryen_opt_level in [0, 1, 2, 3, 4]
        for aot_opt_level in [0, 1, 2, 3]
    ]

    tests = [r for r in tests if r is not None]

    for r in tests:
        print(json.dumps(r))
