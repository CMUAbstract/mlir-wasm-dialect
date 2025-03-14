#! /usr/bin/env python3
import argparse
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict

# Define benchmark categories based on your table of contents
BENCHMARK_CATEGORIES = {
    "Data\nMining": ["covariance", "correlation"],
    "BLAS Routines": ["gemm", "gemver", "gesummv", "symm", "syrk", "syr2k", "trmm"],
    "Linear Algebra Kernels": ["2mm", "3mm", "atax", "bicg", "doitgen", "mvt"],
    "Linear Algebra Solvers": ["cholesky", "durbin", "gramschmidt", "lu", "ludcmp", "trisolv"],
    "Medley": ["deriche", "floyd-marshall", "nussinov"],
    "Stencils": ["adi", "fdtd-2d", "heat-3d", "jacobi-1d", "jacobi-2d", "seidel-2d"],
    "Dynamic\nProgramming": ["floyd-warshall"]

}

# Create reverse mapping for easy category lookup
BENCHMARK_TO_CATEGORY = {}
for category, benchmarks in BENCHMARK_CATEGORIES.items():
    for benchmark in benchmarks:
        BENCHMARK_TO_CATEGORY[benchmark] = category

def parse_data_from_file(filename):
    """Parse the JSON data from the file and extract execution time information."""
    data = []
    
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Extract data from each line (JSON object)
            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    
                    # Extract relevant information
                    benchmark = entry.get('tag')
                    compiler = entry.get('compiler')
                    use_aot = entry.get('use_aot', False)
                    binaryen_opt_level = entry.get('binaryen_opt_level')
                    
                    # Extract execution time from stdout
                    stdout = entry.get('stdout', '')
                    match = re.search(r'\[execution time\] (\d+\.\d+) miliseconds', stdout)
                    if match:
                        execution_time = float(match.group(1))
                        
                        data.append({
                            'benchmark': benchmark,
                            'compiler': compiler,
                            'use_aot': use_aot,
                            'binaryen_opt_level': binaryen_opt_level,
                            'execution_time': execution_time
                        })
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line: {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"Error reading file {filename}: {e}", file=sys.stderr)
        sys.exit(1)
    
    return data

def filter_and_prepare_data(data, use_aot, binaryen_opt_level):
    """Filter the data based on the specified parameters and prepare for plotting."""
    # Convert binaryen_opt_level to integer
    try:
        binaryen_opt_level = int(binaryen_opt_level)
    except ValueError:
        print(f"Error: binaryen_opt_level must be an integer, got: {binaryen_opt_level}", file=sys.stderr)
        sys.exit(1)
    
    # Filter data based on use_aot and binaryen_opt_level
    filtered_data = [d for d in data if d['use_aot'] == use_aot and d['binaryen_opt_level'] == binaryen_opt_level]
    
    # Organize data by benchmark and compiler
    result = {}
    
    for entry in filtered_data:
        benchmark = entry['benchmark']
        compiler = entry['compiler']
        execution_time = entry['execution_time']
        
        if benchmark not in result:
            result[benchmark] = {'llvm': None, 'mlir': None}
        
        result[benchmark][compiler] = execution_time
    
    # Filter out benchmarks without both LLVM and MLIR data
    result = {k: v for k, v in result.items() if v['llvm'] is not None and v['mlir'] is not None}
    
    # Calculate speedup for each benchmark
    for benchmark, values in result.items():
        values['speedup'] = values['llvm'] / values['mlir']
    
    return result

def get_benchmark_category(benchmark):
    """Determine the category of a benchmark based on its name."""
    # Look for exact match
    for key in BENCHMARK_TO_CATEGORY:
        if key in benchmark.lower():
            return BENCHMARK_TO_CATEGORY[key]
    
    # If no match found, return "Others"
    return "Others"

def organize_by_category(data):
    """Organize the benchmark data by category."""
    categorized_data = defaultdict(dict)
    
    for benchmark, values in data.items():
        category = get_benchmark_category(benchmark)
        categorized_data[category][benchmark] = values
    
    return categorized_data

def add_category_labels(ax, group_positions, category_labels):
    for i, (pos, label) in enumerate(zip(group_positions, category_labels)):
        ax.text(pos, ax.get_ylim()[1] * 1.05, label, ha='center', va='bottom', fontsize=8, 
                color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.95, 
                boxstyle='round,pad=0.5', edgecolor='lightgray'))


def plot_data_with_grouped_categories(data, use_aot, binaryen_opt_level, output_file=None, show_speedup=True, normalize=False, only_speedup=False):
    """Create bar plots with benchmarks grouped by category in a single plot."""

    # Academic color scheme - colorblind friendly and prints well in grayscale
    llvm_color = '#0072B2'  # Dark blue
    mlir_color = '#009E73'  # Dark green
    
    # For speedup bars - using a better academic color scheme
    speedup_positive = '#009E73'  # Dark green for MLIR better
    speedup_negative = '#D55E00'  # Dark orange/rust for LLVM better

    # Organize data by category
    categorized_data = organize_by_category(data)
    categories = sorted(categorized_data.keys())
    
    # Calculate overall statistics
    all_speedups = [data[b]['speedup'] for b in data]
    # Calculate geometric mean speedup
    epsilon = 1e-10
    geo_mean_speedup = np.prod(np.array(all_speedups) + epsilon) ** (1.0 / len(all_speedups)) - epsilon
    mlir_wins = len([s for s in all_speedups if s > 1])
    llvm_wins = len([s for s in all_speedups if s < 1])
    
    # Prepare data for plotting with grouped categories
    benchmark_positions = []  # x-positions for the bars
    benchmark_labels = []     # labels for the x-axis
    llvm_times = []           # LLVM execution times
    mlir_times = []           # MLIR execution times
    speedups = []             # Speedup values
    group_positions = []      # midpoint position for each group (for category labels)
    category_labels = []      # category labels
    
    position = 0
    for category in categories:
        benchmarks = sorted(categorized_data[category].keys())
        if not benchmarks:
            continue
            
        category_start = position
        
        for benchmark in benchmarks:
            benchmark_positions.append(position)
            benchmark_labels.append(benchmark)
            llvm_times.append(categorized_data[category][benchmark]['llvm'])
            mlir_times.append(categorized_data[category][benchmark]['mlir'])
            speedups.append(categorized_data[category][benchmark]['speedup'])
            position += 1
            
        # Add extra space between categories
        position += 1
        
        # Calculate the middle position of this category
        category_end = position - 1  # -1 to account for the extra space
        middle = (category_start + category_end) / 2
        group_positions.append(middle)
        category_labels.append(category)
    
    # Create plots
    if only_speedup:
        fig = plt.figure(figsize=(max(15, len(benchmark_labels) * 0.5), 8))
        gs = fig.add_gridspec(1, 1)
        ax2 = fig.add_subplot(gs[0, 0])
    elif show_speedup:
        fig = plt.figure(figsize=(max(15, len(benchmark_labels) * 0.5), 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
    else:
        fig = plt.figure(figsize=(max(15, len(benchmark_labels) * 0.5), 8))
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[0, 0])
    
    # Set figure title
    execution_mode = "Interpreter" if not use_aot else "AOT"
    fig.suptitle(f'Speedup of WAMI over LLVM - {execution_mode}', 
                 fontsize=16, y=0.95)
    
    # Normalize data if requested
    if normalize:
        plot_llvm_times = [1.0 for _ in llvm_times]
        plot_mlir_times = [mlir_times[i] / llvm_times[i] for i in range(len(llvm_times))]
        y_label = 'Normalized Execution Time (relative to LLVM)'
    else:
        plot_llvm_times = llvm_times
        plot_mlir_times = mlir_times
        y_label = 'Execution Time (ms)'
    
    # Plot execution times if not only showing speedup
    if not only_speedup:
        ax = ax1
        width = 0.35
        
        ax.bar([p - width/2 for p in benchmark_positions], plot_llvm_times, width, label='LLVM', color='blue', alpha=0.7)
        ax.bar([p + width/2 for p in benchmark_positions], plot_mlir_times, width, label='MLIR', color='green', alpha=0.7)
        
        # Add vertical lines to separate categories
        for i in range(len(group_positions) - 1):
            midpoint = (benchmark_positions[benchmark_labels.index(sorted(categorized_data[category_labels[i]].keys())[-1])] + 
                       benchmark_positions[benchmark_labels.index(sorted(categorized_data[category_labels[i+1]].keys())[0])]) / 2
            ax.axvline(x=midpoint, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_ylabel(y_label)
        ax.set_title('Execution Time Comparison')
        ax.set_xticks(benchmark_positions)
        ax.set_xticklabels(benchmark_labels, rotation=45, ha='right')
        ax.legend()
        
        add_category_labels(ax, group_positions, category_labels)
    
    # Plot speedup if requested
    if show_speedup or only_speedup:
        ax = ax2
        width = 0.5
        
        # Green for MLIR better (speedup > 1), red for LLVM better (speedup < 1)
        colors = [speedup_positive if s > 1 else speedup_negative for s in speedups]
        ax.bar(benchmark_positions, speedups, width, color=colors)
        ax.axhline(y=1, color='black', linestyle='--')
        
        # Add vertical lines to separate categories
        for i in range(len(group_positions) - 1):
            midpoint = (benchmark_positions[benchmark_labels.index(sorted(categorized_data[category_labels[i]].keys())[-1])] + 
                       benchmark_positions[benchmark_labels.index(sorted(categorized_data[category_labels[i+1]].keys())[0])]) / 2
            ax.axvline(x=midpoint, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_ylabel('Speedup (LLVM/WAMI)')
        # ax.set_title('Speedup Ratio (> 1 means MLIR is better)')
        ax.set_xticks(benchmark_positions)
        ax.set_xticklabels(benchmark_labels, rotation=45, ha='right')
        
        # Add category labels above the plot
        add_category_labels(ax, group_positions, category_labels)
    
    # Add summary text
    summary_text = (f"Geometric Mean Speedup (LLVM/MLIR): {geo_mean_speedup:.3f}\n"
                   f"Benchmarks where MLIR outperforms LLVM: {mlir_wins} out of {len(all_speedups)}\n"
                   f"Benchmarks where LLVM outperforms MLIR: {llvm_wins} out of {len(all_speedups)}")
    
    # Print summary to stdout
    print("\nSummary Statistics:")
    print(summary_text)
    
    # Add a text box with the summary at the bottom of the figure
    #fig.text(0.5, 0.01, summary_text, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot LLVM vs MLIR performance comparison grouped by category')
    parser.add_argument('filename', help='Input data file')
    parser.add_argument('--aot', dest='use_aot', action='store_true', help='Use AOT mode (default: interpreter mode)')
    parser.add_argument('--binaryen-opt-level', type=int, choices=[0, 2, 4], default=0,
                       help='Binaryen optimization level (0, 2, or 4, default: 0)')
    parser.add_argument('--normalize', action='store_true', help='Normalize execution times')
    parser.add_argument('--no-speedup', dest='show_speedup', action='store_false', 
                        help='Disable speedup plot')
    parser.add_argument('--only-speedup', action='store_true',
                        help='Show only the speedup plot')
    parser.add_argument('-o', '--output', help='Output file for the plot (optional)')
    
    # Set defaults for new arguments
    parser.set_defaults(show_speedup=True)
    
    args = parser.parse_args()
    
    # Parse data from the file
    data = parse_data_from_file(args.filename)
    
    # Filter and prepare data based on the specified parameters
    filtered_data = filter_and_prepare_data(data, args.use_aot, args.binaryen_opt_level)
    
    # Plot the data with grouped categories in a single plot
    plot_data_with_grouped_categories(
        filtered_data, args.use_aot, args.binaryen_opt_level, args.output,
        args.show_speedup, args.normalize, args.only_speedup
    )

if __name__ == "__main__":
    main()