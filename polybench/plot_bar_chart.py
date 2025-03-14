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
    "BLAS\nRoutines": ["gemm", "gemver", "gesummv", "symm", "syrk", "syr2k", "trmm"],
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
    """Add category labels above the plot."""
    for i, (pos, label) in enumerate(zip(group_positions, category_labels)):
        ax.text(pos, ax.get_ylim()[1] * 1.05, label, ha='center', va='bottom', fontsize=8, 
                color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.95, 
                boxstyle='round,pad=0.5', edgecolor='lightgray'))

def prepare_plot_data(data):
    """Prepare data for plotting by organizing by category."""
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
    
    return {
        'categorized_data': categorized_data,
        'benchmark_positions': benchmark_positions,
        'benchmark_labels': benchmark_labels,
        'llvm_times': llvm_times,
        'mlir_times': mlir_times,
        'speedups': speedups,
        'group_positions': group_positions,
        'category_labels': category_labels,
        'statistics': {
            'geo_mean_speedup': geo_mean_speedup,
            'mlir_wins': mlir_wins,
            'llvm_wins': llvm_wins,
            'total_benchmarks': len(all_speedups)
        }
    }

# Modification to remove hatching from bars

def plot_execution_time(data, use_aot, binaryen_opt_level, output_file=None, normalize=False):
    """Plot execution time comparison chart."""
    # Academic color scheme
    llvm_color = '#0072B2'  # Dark blue
    mlir_color = '#009E73'  # Dark green
    
    # Prepare data
    plot_data = prepare_plot_data(data)
    benchmark_positions = plot_data['benchmark_positions']
    benchmark_labels = plot_data['benchmark_labels']
    llvm_times = plot_data['llvm_times']
    mlir_times = plot_data['mlir_times']
    group_positions = plot_data['group_positions']
    category_labels = plot_data['category_labels']
    categorized_data = plot_data['categorized_data']
    
    # Normalize data if requested
    if normalize:
        plot_llvm_times = [1.0 for _ in llvm_times]
        plot_mlir_times = [mlir_times[i] / llvm_times[i] for i in range(len(llvm_times))]
        y_label = 'Normalized Execution Time (relative to LLVM)'
    else:
        plot_llvm_times = llvm_times
        plot_mlir_times = mlir_times
        y_label = 'Execution Time (ms)'
    
    # Create figure
    fig = plt.figure(figsize=(max(15, len(benchmark_labels) * 0.5), 8))
    ax = fig.add_subplot(1, 1, 1)
    
    # Set figure title
    execution_mode = "Interpreter" if not use_aot else "AOT"
    fig.suptitle(f'Execution Time Comparison - {execution_mode}', 
                 fontsize=16, y=0.95)
    
    # Plot execution times
    width = 0.35
    
    # Add bars WITHOUT hatching
    llvm_bars = ax.bar([p - width/2 for p in benchmark_positions], plot_llvm_times, width, 
                      label='LLVM', color=llvm_color, alpha=0.85)
    mlir_bars = ax.bar([p + width/2 for p in benchmark_positions], plot_mlir_times, width, 
                       label='WAMI', color=mlir_color, alpha=0.85)
    
    # Add vertical lines to separate categories
    for i in range(len(group_positions) - 1):
        midpoint = (benchmark_positions[benchmark_labels.index(sorted(categorized_data[category_labels[i]].keys())[-1])] + 
                   benchmark_positions[benchmark_labels.index(sorted(categorized_data[category_labels[i+1]].keys())[0])]) / 2
        ax.axvline(x=midpoint, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Add gridlines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_xticks(benchmark_positions)
    ax.set_xticklabels(benchmark_labels, rotation=45, ha='right')
    ax.legend(frameon=True, framealpha=0.95)
    
    add_category_labels(ax, group_positions, category_labels)
    
    # Add summary statistics to stdout
    stats = plot_data['statistics']
    summary_text = (f"Geometric Mean Speedup (LLVM/WAMI): {stats['geo_mean_speedup']:.3f}\n"
                   f"Benchmarks where WAMI outperforms LLVM: {stats['mlir_wins']} out of {stats['total_benchmarks']}\n"
                   f"Benchmarks where LLVM outperforms WAMI: {stats['llvm_wins']} out of {stats['total_benchmarks']}")
    
    print("\nSummary Statistics:")
    print(summary_text)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    
    if output_file:
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        # Save PDF version if not already PDF
        if not output_file.endswith('.pdf'):
            pdf_file = f"{output_file.split('.')[0]}.pdf"
            plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
            print(f"PDF version saved to {pdf_file}")
    
    plt.show()

def plot_speedup(data, use_aot, binaryen_opt_level, output_file=None):
    """Plot speedup comparison chart with a more compact vertical scale."""
    # Academic color scheme for speedup
    speedup_positive = '#009E73'  # Dark green for MLIR better
    speedup_negative = '#D55E00'  # Dark orange/rust for LLVM better
    
    # Prepare data
    plot_data = prepare_plot_data(data)
    benchmark_positions = plot_data['benchmark_positions']
    benchmark_labels = plot_data['benchmark_labels']
    speedups = plot_data['speedups']
    group_positions = plot_data['group_positions']
    category_labels = plot_data['category_labels']
    categorized_data = plot_data['categorized_data']
    
    # Create figure
    fig = plt.figure(figsize=(max(15, len(benchmark_labels) * 0.5), 6))  # Reduced height
    ax = fig.add_subplot(1, 1, 1)
    
    # Set figure title
    execution_mode = "Interpreter" if not use_aot else "AOT"
    fig.suptitle(f'Speedup of WAMI over LLVM - {execution_mode}', 
                 fontsize=16, y=0.95)
    
    # Plot speedup bars
    width = 0.5
    
    # Create bars WITHOUT hatching
    for i, (pos, speedup) in enumerate(zip(benchmark_positions, speedups)):
        color = speedup_positive if speedup > 1 else speedup_negative
        ax.bar(pos, speedup, width, color=color, alpha=0.85)
    
    # Find the min and max speedup to set appropriate y-axis limits
    min_speedup = min(speedups)
    max_speedup = max(speedups)
    
    # Set compact y-axis limits
    # Find rounded values that give us some padding but keep the scale compact
    # Ensure 1.0 is always in the middle of the visible range for good reference
    padding = 0.05  # Adjustable padding
    
    # Calculate distance from 1.0
    distance_below = 1.0 - min_speedup
    distance_above = max_speedup - 1.0
    max_distance = max(distance_below, distance_above)
    
    # Set symmetric limits around 1.0 with a small padding
    y_min = 1.0 - max_distance - padding
    y_max = 1.0 + max_distance + padding
    
    # If we have a very narrow range, force a minimum range
    if y_max - y_min < 0.5:
        y_min = max(0.7, 1.0 - 0.25)  # Don't go below 0.7
        y_max = min(1.4, 1.0 + 0.25)  # Default range of +/- 0.25 around 1.0
    
    ax.set_ylim(y_min, y_max)
    
    # Add reference line
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
    
    # Add vertical lines to separate categories
    for i in range(len(group_positions) - 1):
        midpoint = (benchmark_positions[benchmark_labels.index(sorted(categorized_data[category_labels[i]].keys())[-1])] + 
                   benchmark_positions[benchmark_labels.index(sorted(categorized_data[category_labels[i+1]].keys())[0])]) / 2
        ax.axvline(x=midpoint, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Add gridlines with more frequent y ticks
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set more y ticks to better show small differences
    y_ticks = np.arange(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10 + 0.01, 0.05)
    ax.set_yticks(y_ticks)
    
    # Format y tick labels to show 2 decimal places
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    ax.set_ylabel('Speedup (LLVM/WAMI)', fontsize=11)
    ax.set_xticks(benchmark_positions)
    ax.set_xticklabels(benchmark_labels, rotation=45, ha='right')
    
    # Add legend for speedup plot WITHOUT hatching
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=speedup_positive, alpha=0.85, label='WAMI faster'),
        Patch(facecolor=speedup_negative, alpha=0.85, label='LLVM faster')
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True, framealpha=0.95)
    
    add_category_labels(ax, group_positions, category_labels)
    
    # Add summary statistics to stdout
    stats = plot_data['statistics']
    summary_text = (f"Geometric Mean Speedup (LLVM/WAMI): {stats['geo_mean_speedup']:.3f}\n"
                   f"Benchmarks where WAMI outperforms LLVM: {stats['mlir_wins']} out of {stats['total_benchmarks']}\n"
                   f"Benchmarks where LLVM outperforms WAMI: {stats['llvm_wins']} out of {stats['total_benchmarks']}")
    
    print("\nSummary Statistics:")
    print(summary_text)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    
    if output_file:
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        # Save PDF version if not already PDF
        if not output_file.endswith('.pdf'):
            pdf_file = f"{output_file.split('.')[0]}.pdf"
            plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
            print(f"PDF version saved to {pdf_file}")
    
    plt.show()

def plot_percentage_improvement(data, use_aot, binaryen_opt_level, output_file=None):
    """
    Plot percentage improvement chart showing how much better/worse WAMI performs compared to LLVM.
    Positive percentages indicate WAMI is faster, negative percentages indicate LLVM is faster.
    """
    # Color scheme for improvements
    improvement_positive = '#009E73'  # Dark green for WAMI better
    improvement_negative = '#D55E00'  # Dark orange/rust for LLVM better
    
    # Prepare data
    plot_data = prepare_plot_data(data)
    benchmark_positions = plot_data['benchmark_positions']
    benchmark_labels = plot_data['benchmark_labels']
    speedups = plot_data['speedups']
    group_positions = plot_data['group_positions']
    category_labels = plot_data['category_labels']
    categorized_data = plot_data['categorized_data']
    
    # Calculate percentage improvements from speedups
    # Formula: (speedup - 1) * 100% for WAMI faster (speedup > 1)
    # Formula: (1 - 1/speedup) * 100% for LLVM faster (speedup < 1)
    percentage_improvements = []
    for speedup in speedups:
        if speedup >= 1:
            # WAMI is faster - calculate how much faster as percentage
            percentage = (speedup - 1) * 100
        else:
            # LLVM is faster - calculate how much faster as a negative percentage
            percentage = (1 - 1/speedup) * 100
        percentage_improvements.append(percentage)
    
    # Create figure
    fig = plt.figure(figsize=(max(15, len(benchmark_labels) * 0.5), 6))
    ax = fig.add_subplot(1, 1, 1)
    
    # Set figure title
    execution_mode = "Interpreter" if not use_aot else "AOT"
    fig.suptitle(f'Percentage Improvement of WAMI vs LLVM - {execution_mode}', 
                 fontsize=16, y=0.95)
    
    # Plot percentage improvement bars
    width = 0.5
    
    # Create bars with colors based on whether improvement is positive or negative
    for i, (pos, percentage) in enumerate(zip(benchmark_positions, percentage_improvements)):
        color = improvement_positive if percentage >= 0 else improvement_negative
        ax.bar(pos, percentage, width, color=color, alpha=0.85)
    
    # Find reasonable y-axis limits
    min_percentage = min(percentage_improvements)
    max_percentage = max(percentage_improvements)
    padding = 5  # 5% padding
    
    # Set y limits with padding
    y_min = min_percentage - padding if min_percentage < 0 else -padding
    y_max = max_percentage + padding if max_percentage > 0 else padding
    
    # Ensure zero is visible
    if y_min > 0:
        y_min = -padding
    if y_max < 0:
        y_max = padding
        
    # Ensure there's some reasonable separation from zero for small values
    if abs(y_max - y_min) < 20:
        y_min = min(-10, y_min)
        y_max = max(10, y_max)
    
    ax.set_ylim(y_min, y_max)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Add vertical lines to separate categories
    for i in range(len(group_positions) - 1):
        midpoint = (benchmark_positions[benchmark_labels.index(sorted(categorized_data[category_labels[i]].keys())[-1])] + 
                   benchmark_positions[benchmark_labels.index(sorted(categorized_data[category_labels[i+1]].keys())[0])]) / 2
        ax.axvline(x=midpoint, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Add gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    ax.set_ylabel('Percentage Improvement (%)', fontsize=11)
    ax.set_xticks(benchmark_positions)
    ax.set_xticklabels(benchmark_labels, rotation=45, ha='right')
    
    # Add percentage sign to y-axis tick labels
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=improvement_positive, alpha=0.85, label='WAMI faster'),
        Patch(facecolor=improvement_negative, alpha=0.85, label='LLVM faster')
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True, framealpha=0.95)
    
    add_category_labels(ax, group_positions, category_labels)
    
    # Calculate average improvement
    avg_improvement = sum(percentage_improvements) / len(percentage_improvements)
    
    # Add summary statistics to stdout
    stats = plot_data['statistics']
    summary_text = (
        f"Average Percentage Improvement: {avg_improvement:.2f}%\n"
        f"Geometric Mean Speedup (LLVM/WAMI): {stats['geo_mean_speedup']:.3f}\n"
        f"Benchmarks where WAMI outperforms LLVM: {stats['mlir_wins']} out of {stats['total_benchmarks']}\n"
        f"Benchmarks where LLVM outperforms WAMI: {stats['llvm_wins']} out of {stats['total_benchmarks']}"
    )
    
    print("\nSummary Statistics:")
    print(summary_text)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    
    if output_file:
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        # Save PDF version if not already PDF
        if not output_file.endswith('.pdf'):
            pdf_file = f"{output_file.split('.')[0]}.pdf"
            plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
            print(f"PDF version saved to {pdf_file}")
    
    plt.show()

# Update main function to include the new chart type
def main():
    parser = argparse.ArgumentParser(description='Plot LLVM vs WAMI performance comparison grouped by category')
    parser.add_argument('filename', help='Input data file')
    parser.add_argument('--aot', dest='use_aot', action='store_true', help='Use AOT mode (default: interpreter mode)')
    parser.add_argument('--binaryen-opt-level', type=int, choices=[0, 2, 4], default=0,
                       help='Binaryen optimization level (0, 2, or 4, default: 0)')
    parser.add_argument('--normalize', action='store_true', help='Normalize execution times (only applies to time chart)')
    parser.add_argument('--chart-type', choices=['time', 'speedup', 'percentage'], required=True,
                        help='Type of chart to display: execution time, speedup, or percentage improvement')
    parser.add_argument('-o', '--output', help='Output file for the plot (optional)')
    
    args = parser.parse_args()
    
    # Parse data from the file
    data = parse_data_from_file(args.filename)
    
    # Filter and prepare data based on the specified parameters
    filtered_data = filter_and_prepare_data(data, args.use_aot, args.binaryen_opt_level)
    
    # Plot the appropriate chart type
    if args.chart_type == 'time':
        plot_execution_time(
            filtered_data, args.use_aot, args.binaryen_opt_level,
            output_file=args.output, normalize=args.normalize
        )
    elif args.chart_type == 'speedup':
        plot_speedup(
            filtered_data, args.use_aot, args.binaryen_opt_level,
            output_file=args.output
        )
    elif args.chart_type == 'percentage':
        plot_percentage_improvement(
            filtered_data, args.use_aot, args.binaryen_opt_level,
            output_file=args.output
        )
    else:
        print("Invalid chart type. Please use 'time', 'speedup', or 'percentage'.")

if __name__ == "__main__":
    main()
