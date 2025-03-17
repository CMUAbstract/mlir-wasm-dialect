#! /usr/bin/env python3
import argparse
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict

# Define benchmark categories with line breaks for long names
BENCHMARK_CATEGORIES = {
    "Data\nMining": ["covariance", "correlation"],
    "BLAS Routines": ["gemm", "gemver", "gesummv", "symm", "syrk", "syr2k", "trmm"],
    "Linear Algebra\nKernels": ["2mm", "3mm", "atax", "bicg", "doitgen", "mvt"],
    "Linear Algebra\nSolvers": ["cholesky", "durbin", "gramschmidt", "lu", "ludcmp", "trisolv"],
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
    for key in BENCHMARK_TO_CATEGORY:
        if key in benchmark.lower():
            return BENCHMARK_TO_CATEGORY[key]
    return "Others"

def organize_by_category(data):
    """Organize the benchmark data by category."""
    categorized_data = defaultdict(dict)
    for benchmark, values in data.items():
        category = get_benchmark_category(benchmark)
        categorized_data[category][benchmark] = values
    return categorized_data

def prepare_plot_data(data, bar_spacing=0.6):
    """
    Prepare data for plotting by organizing by category.
    Also calculate the correct center position of each category group.
    """
    # Organize data by category
    categorized_data = organize_by_category(data)
    categories = sorted(categorized_data.keys())
    
    # Calculate overall statistics
    all_speedups = [data[b]['speedup'] for b in data]
    epsilon = 1e-10
    geo_mean_speedup = np.prod(np.array(all_speedups) + epsilon) ** (1.0 / len(all_speedups)) - epsilon
    mlir_wins = len([s for s in all_speedups if s > 1])
    llvm_wins = len([s for s in all_speedups if s < 1])
    
    benchmark_positions = []
    benchmark_labels = []
    llvm_times = []
    mlir_times = []
    speedups = []
    group_positions = []
    category_labels = []
    category_start_end = {}
    
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
            position += bar_spacing
        
        category_end = position - bar_spacing
        category_start_end[category] = (category_start, category_end)
        middle = (category_start + category_end) / 2
        group_positions.append(middle)
        category_labels.append(category)
        
        position += bar_spacing * 1.5
    
    return {
        'categorized_data': categorized_data,
        'benchmark_positions': benchmark_positions,
        'benchmark_labels': benchmark_labels,
        'llvm_times': llvm_times,
        'mlir_times': mlir_times,
        'speedups': speedups,
        'group_positions': group_positions,
        'category_labels': category_labels,
        'category_start_end': category_start_end,
        'statistics': {
            'geo_mean_speedup': geo_mean_speedup,
            'mlir_wins': mlir_wins,
            'llvm_wins': llvm_wins,
            'total_benchmarks': len(all_speedups)
        }
    }

def add_category_labels(fig, ax, group_positions, category_labels):
    """Add category labels at the bottom of the plot using a separate axis."""
    main_pos = ax.get_position()
    label_height = 0.08
    
    # Create a new axis for the category labels below the main plot
    label_ax = fig.add_axes([
        main_pos.x0, 
        main_pos.y0 - 0.2, 
        main_pos.width, 
        label_height
    ])
    label_ax.axis('off')
    
    # Match x-limits with main axis
    label_ax.set_xlim(ax.get_xlim())
    
    for pos, label in zip(group_positions, category_labels):
        label_ax.text(
            pos, 0.5, label,
            ha='center', va='center', fontsize=12, fontweight='bold',
            rotation=45
        )

def plot_speedup(data, use_aot, binaryen_opt_level, output_file=None):
    """Plot speedup comparison chart with wider bars and properly centered category labels."""
    speedup_positive = '#009E73'  # Dark green
    speedup_negative = '#D55E00'  # Dark orange
    
    plot_data = prepare_plot_data(data, bar_spacing=0.6)
    benchmark_positions = plot_data['benchmark_positions']
    benchmark_labels = plot_data['benchmark_labels']
    speedups = plot_data['speedups']
    group_positions = plot_data['group_positions']
    category_labels = plot_data['category_labels']
    category_start_end = plot_data['category_start_end']
    
    fig = plt.figure(figsize=(max(15, len(benchmark_labels) * 0.5), 7))
    ax = fig.add_subplot(1, 1, 1)
    
    execution_mode = "Interpreter" if not use_aot else "AOT"
    fig.suptitle(f'Speedup of WAMI over LLVM - {execution_mode}', fontsize=16, y=0.95)
    
    width = 0.5
    
    # Plot the bars
    for pos, speedup in zip(benchmark_positions, speedups):
        color = speedup_positive if speedup > 1 else speedup_negative
        ax.bar(pos, speedup, width, color=color, alpha=0.85)
    
    # Determine y-axis limits
    min_speedup = min(speedups)
    max_speedup = max(speedups)
    padding = 0.05
    distance_below = 1.0 - min_speedup
    distance_above = max_speedup - 1.0
    max_distance = max(distance_below, distance_above)
    y_min = 1.0 - max_distance - padding
    y_max = 1.0 + max_distance + padding
    
    if y_max - y_min < 0.5:
        y_min = max(0.7, 1.0 - 0.25)
        y_max = min(1.4, 1.0 + 0.25)
    
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
    
    # Vertical lines between categories
    for i in range(len(category_labels) - 1):
        category = category_labels[i]
        next_category = category_labels[i + 1]
        _, current_end = category_start_end[category]
        next_start, _ = category_start_end[next_category]
        midpoint = (current_end + next_start) / 2
        ax.axvline(x=midpoint, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Add gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Ticks
    y_ticks = np.arange(
        np.floor(y_min * 10) / 10, 
        np.ceil(y_max * 10) / 10 + 0.01, 
        0.05
    )
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.set_ylabel('Speedup (LLVM/WAMI)', fontsize=11)
    ax.set_xticks(benchmark_positions)
    ax.set_xticklabels(benchmark_labels, rotation=45, ha='right')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=speedup_positive, alpha=0.85, label='WAMI faster'),
        Patch(facecolor=speedup_negative, alpha=0.85, label='LLVM faster')
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True, framealpha=0.95)
    
    # - Reduce left/right margin
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.25, top=0.90)
    # Category labels at bottom
    add_category_labels(fig, ax, group_positions, category_labels)
    
    # Print summary stats
    stats = plot_data['statistics']
    summary_text = (
        f"Geometric Mean Speedup (LLVM/WAMI): {stats['geo_mean_speedup']:.3f}\n"
        f"Benchmarks where WAMI outperforms LLVM: {stats['mlir_wins']} out of {stats['total_benchmarks']}\n"
        f"Benchmarks where LLVM outperforms WAMI: {stats['llvm_wins']} out of {stats['total_benchmarks']}"
    )
    print("\nSummary Statistics:")
    print(summary_text)
    
    # Adjust margins:
    # - Keep bottom margin for the category labels axis
    
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
        
        # Optional PDF
        if not output_file.endswith('.pdf'):
            pdf_file = f"{output_file.split('.')[0]}.pdf"
            plt.savefig(pdf_file, format='pdf', dpi=300)
            print(f"PDF version saved to {pdf_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot LLVM vs WAMI performance comparison grouped by category')
    parser.add_argument('filename', help='Input data file')
    parser.add_argument('--aot', dest='use_aot', action='store_true',
                        help='Use AOT mode (default: interpreter mode)')
    parser.add_argument('--binaryen-opt-level', type=int, choices=[0, 2, 4], default=0,
                        help='Binaryen optimization level (0, 2, or 4, default: 0)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize execution times (only applies to time chart)')
    parser.add_argument('-o', '--output', help='Output file for the plot (optional)')
    
    args = parser.parse_args()
    
    data = parse_data_from_file(args.filename)
    filtered_data = filter_and_prepare_data(data, args.use_aot, args.binaryen_opt_level)
    
    plot_speedup(filtered_data, args.use_aot, args.binaryen_opt_level, output_file=args.output)

if __name__ == "__main__":
    main()
