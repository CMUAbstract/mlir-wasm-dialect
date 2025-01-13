#!/usr/bin/env python3

import sys
import csv
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot data from CSV with optional filtering and normalization."
    )
    parser.add_argument("csv_file", help="Path to the CSV file.")
    parser.add_argument(
        "--include", 
        nargs="+", 
        default=None, 
        help="List of test case names (benchmarks) to be shown in the plots. If omitted, show all."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize MLIR times and LLVM times by LLVM-based time for each benchmark."
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Prefix for saving four image files, e.g. '--output myplots' will create "
            "'myplots_int_no_binaryen.png', 'myplots_int_binaryen.png', etc. If omitted, the plots are displayed."
        )
    )
    return parser.parse_args()

##########################################
# Global settings for nicer publication. #
##########################################

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
    "font.family": "serif"
})

# Custom color palette for bars
llvm_color = "#1f77b4"  # Blue
mlir_color = "#ff7f0e"  # Orange

##########################################
# Dictionaries for storing parsed data.  #
##########################################

llvm_interp_no_binaryen = {}
llvm_interp_binaryen    = {}
mlir_interp_no_binaryen = {}
mlir_interp_binaryen    = {}

llvm_aot_no_binaryen    = {}
llvm_aot_binaryen       = {}
mlir_aot_no_binaryen    = {}
mlir_aot_binaryen       = {}

##########################################
# Convert string to float or None.       #
##########################################

def parse_time(value):
    if not value:
        return None
    v = value.strip().lower()
    invalid_values = [
        "fixme: malloc failed!",
        "#value!",
        "#div/0!",
        "this contains alloca. not sure yet how to handle this."
    ]
    if any(iv in v for iv in invalid_values):
        return None
    try:
        return float(v)
    except ValueError:
        return None

##########################################
# Read the CSV and populate dictionaries #
##########################################

def read_csv(csv_file):
    benchmarks = []
    last_benchmark = None

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue

            compiler_field = row[2].strip().lower()
            if compiler_field not in ["llvm (-o3)", "mlir"]:
                continue

            benchmark_name = row[1].strip()
            if benchmark_name:
                last_benchmark = benchmark_name
            elif last_benchmark:
                benchmark_name = last_benchmark
            else:
                # no valid benchmark name
                continue

            if benchmark_name not in benchmarks:
                benchmarks.append(benchmark_name)

            interp_no_binaryen = parse_time(row[3])
            interp_binaryen    = parse_time(row[4])
            aot_no_binaryen    = parse_time(row[5])
            aot_binaryen       = parse_time(row[6])

            if compiler_field == "llvm (-o3)":
                llvm_interp_no_binaryen[benchmark_name] = interp_no_binaryen
                llvm_interp_binaryen[benchmark_name]    = interp_binaryen
                llvm_aot_no_binaryen[benchmark_name]    = aot_no_binaryen
                llvm_aot_binaryen[benchmark_name]       = aot_binaryen
            else:  # MLIR
                mlir_interp_no_binaryen[benchmark_name] = interp_no_binaryen
                mlir_interp_binaryen[benchmark_name]    = interp_binaryen
                mlir_aot_no_binaryen[benchmark_name]    = aot_no_binaryen
                mlir_aot_binaryen[benchmark_name]       = aot_binaryen

    return benchmarks

##########################################
# Optionally filter benchmarks.          #
##########################################

def maybe_filter_benchmarks(benchmarks, include_list):
    if not include_list:
        return benchmarks
    return [b for b in benchmarks if b in include_list]

##########################################
# Optionally normalize MLIR vs. LLVM.    #
##########################################

def normalize_pair(llvm_data, mlir_data):
    keys_in_both = set(llvm_data.keys()).intersection(set(mlir_data.keys()))
    for b in keys_in_both:
        llvm_val = llvm_data[b]
        mlir_val = mlir_data[b]
        if llvm_val is not None and llvm_val != 0:
            if mlir_val is not None:
                mlir_data[b] = mlir_val / llvm_val
            llvm_data[b] = 1.0

def maybe_normalize(normalize_flag):
    if not normalize_flag:
        return
    normalize_pair(llvm_interp_no_binaryen, mlir_interp_no_binaryen)
    normalize_pair(llvm_interp_binaryen,    mlir_interp_binaryen)
    normalize_pair(llvm_aot_no_binaryen,    mlir_aot_no_binaryen)
    normalize_pair(llvm_aot_binaryen,       mlir_aot_binaryen)

##########################################
# Create a single bar chart figure.      #
##########################################

def plot_and_save_or_show(
    benchmarks, llvm_data, mlir_data, title, 
    filename=None, normalize=False
):
    """
    Creates a figure, draws bars for the given dictionaries,
    saves to 'filename' if given, otherwise shows the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(benchmarks))
    width = 0.4

    llvm_times = []
    mlir_times = []
    for b in benchmarks:
        llvm_val = llvm_data.get(b, 0.0) or 0.0
        mlir_val = mlir_data.get(b, 0.0) or 0.0
        llvm_times.append(llvm_val)
        mlir_times.append(mlir_val)

    # Plot bars
    ax.bar(
        x - width/2, llvm_times, width,
        label="LLVM", color=llvm_color, 
        edgecolor="black", linewidth=0.7
    )
    ax.bar(
        x + width/2, mlir_times, width,
        label="MLIR", color=mlir_color, 
        edgecolor="black", linewidth=0.7
    )

    # Title, axis labels
    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right", rotation_mode="anchor")
    if normalize:
        ax.set_ylabel("Relative to LLVM")
    else:
        ax.set_ylabel("Execution Time (ms)")

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")
        plt.close(fig)
    else:
        plt.show()

##########################################
# Main function                          #
##########################################

def main():
    args = parse_args()
    all_benchmarks = read_csv(args.csv_file)
    filtered_benchmarks = maybe_filter_benchmarks(all_benchmarks, args.include)
    maybe_normalize(args.normalize)

    # If the user gave an --output prefix, we'll create 4 filenames.
    # Otherwise, we will display the figures interactively.
    if args.output:
        prefix = args.output
    else:
        prefix = None  # meaning "no file saving"

    # 1) Interpreter (w/o binaryen)
    title_no_bin_int = "Interpreter (w/o binaryen)"
    f1_name = f"{prefix}_int_no_binaryen.png" if prefix else None
    plot_and_save_or_show(
        filtered_benchmarks,
        llvm_interp_no_binaryen,
        mlir_interp_no_binaryen,
        title_no_bin_int,
        filename=f1_name,
        normalize=args.normalize
    )

    # 2) Interpreter (w/ binaryen)
    title_bin_int = "Interpreter (w/ binaryen)"
    f2_name = f"{prefix}_int_binaryen.png" if prefix else None
    plot_and_save_or_show(
        filtered_benchmarks,
        llvm_interp_binaryen,
        mlir_interp_binaryen,
        title_bin_int,
        filename=f2_name,
        normalize=args.normalize
    )

    # 3) AOT (-O3) (w/o binaryen)
    title_no_bin_aot = "AOT (-O3) (w/o binaryen)"
    f3_name = f"{prefix}_aot_no_binaryen.png" if prefix else None
    plot_and_save_or_show(
        filtered_benchmarks,
        llvm_aot_no_binaryen,
        llvm_aot_binaryen,
        title_no_bin_aot,
        filename=f3_name,
        normalize=args.normalize
    )

    # 4) AOT (-O3) (w/ binaryen)
    title_bin_aot = "AOT (-O3) (w/ binaryen)"
    f4_name = f"{prefix}_aot_binaryen.png" if prefix else None
    plot_and_save_or_show(
        filtered_benchmarks,
        llvm_aot_bin,
        mlir_aot_binaryen,
        title_bin_aot,
        filename=f4_name,
        normalize=args.normalize
    )

if __name__ == "__main__":
    main()