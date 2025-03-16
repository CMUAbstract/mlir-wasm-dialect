#!/usr/bin/env python3
import json

FILENAME = {
    "2mm": "2mm.mlir",
    "3mm": "3mm.mlir",
    "adi": "adi.mlir",
    "atax": "atax.mlir",
    "bicg": "bicg.mlir",
    "cholesky": "cholesky.mlir",
    "correlation": "correlation.mlir",
    "covariance": "covariance.mlir",
    "doitgen": "doitgen.mlir",
    "durbin": "durbin.mlir",
    "fdtd-2d": "fdtd-2d.mlir",
    "floyd-warshall": "floyd-warshall.mlir",
    "gemm": "gemm.mlir",
    "gemver": "gemver.mlir",
    "gesummv": "gesummv.mlir",
    "gramschmidt": "gramschmidt.mlir",
    "heat-3d": "heat-3d.mlir",
    "jacobi-1d": "jacobi-1d.mlir",
    "jacobi-2d": "jacobi-2d.mlir",
    "lu": "lu.mlir",
    "ludcmp": "ludcmp.mlir",
    "mvt": "mvt.mlir",
    "nussinov": "nussinov.mlir",
    "seidel-2d": "seidel-2d.mlir",
    "symm": "symm.mlir",
    "syr2k": "syr2k.mlir",
    "syrk": "syrk.mlir",
    "trisolv": "trisolv.mlir",
    "trmm": "trmm.mlir",
}


def cmd(
    device: str,
    size: str,
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

    if device == "mcu":
        aot_str = (
            f"-- --opt-level={aot_opt_level} --target=thumbv7em --target-abi=eabihf --cpu=cortex-m4"
            if use_aot
            else ""
        )
    else:
        aot_str = f"-- --opt-level={aot_opt_level} --target=aarch64v8 --cpu=apple-m1"

    file_name = FILENAME[tag]

    if device == "mcu":
        cmd_parts = [
            'echo "cd .. && ./run-mcu.sh',
            f"--device={device}",
            f"polybench/{size}/{file_name}",
            f"--compiler={compiler}",
            f"--llvm-opt-flags={llvm_opt_flags}" if compiler == "llvm" else "",
            f"--binaryen-opt-flags={binaryen_opt_flags}",
            f'--use-aot={"true" if use_aot else "false"}',
            "--silent",
            aot_str,
            '" | pipenv run ./measure.py',
        ]
    elif device == "local":
        cmd_parts = [
            "cd .. && ./run-mcu.sh",
            f"--device={device}",
            f"polybench/{size}/{file_name}",
            f"--compiler={compiler}",
            f"--llvm-opt-flags={llvm_opt_flags}" if compiler == "llvm" else "",
            f"--binaryen-opt-flags={binaryen_opt_flags}",
            f'--use-aot={"true" if use_aot else "false"}',
            aot_str,
        ]
    else:
        raise ValueError(f"Invalid device: {device}")

    cmd_template = " ".join(filter(bool, cmd_parts))
    return cmd_template


def make_row(
    device: str,
    size: str,
    tag: str,
    compiler: str,
    use_aot: bool,
    *,
    llvm_opt_level: int,
    binaryen_opt_level: int,
    aot_opt_level=-1,
) -> dict:
    row = {
        "device": device,
        "size": size,
        "tag": tag,
        "cmd": cmd(
            device,
            size,
            tag,
            compiler,
            use_aot,
            llvm_opt_level=llvm_opt_level,
            binaryen_opt_level=binaryen_opt_level,
            aot_opt_level=aot_opt_level,
        ),
        "compiler": compiler,
        "use_aot": use_aot,
        "binaryen_opt_level": binaryen_opt_level,
    }
    if compiler == "llvm":
        row["llvm_opt_level"] = llvm_opt_level
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
        "2mm",
        "3mm",
        "adi",
        "atax",
        "bicg",
        "cholesky",
        "correlation",
        "covariance",
        "doitgen",
        "durbin",
        "fdtd-2d",
        "floyd-warshall",
        "gemm",
        "gemver",
        "gesummv",
        "gramschmidt",
        "heat-3d",
        "jacobi-1d",
        "jacobi-2d",
        "lu",
        "ludcmp",
        "mvt",
        "nussinov",
        "seidel-2d",
        "symm",
        "syr2k",
        "syrk",
        "trisolv",
        "trmm",
    ]

    devices = ["local", "mcu"]
    sizes = ["small", "medium", "extralarge"]

    tests = [
        filter_unique(
            make_row(
                device,
                size,
                tag,
                compiler,
                use_aot,
                llvm_opt_level=llvm_opt_level,
                binaryen_opt_level=binaryen_opt_level,
                aot_opt_level=aot_opt_level,
            )
        )
        for device in devices
        for tag in tags
        for size in sizes
        for compiler in ["mlir", "llvm"]
        for use_aot in [True, False]
        for llvm_opt_level in [0, 1, 2, 3]
        for binaryen_opt_level in [0, 1, 2, 3, 4]
        for aot_opt_level in [0, 1, 2, 3]
    ]

    tests = [r for r in tests if r is not None]

    for r in tests:
        print(json.dumps(r))
