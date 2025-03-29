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
    "Data Mining": ["covariance", "correlation"],
    "BLAS Routines": ["gemm", "gemver", "gesummv", "symm", "syrk", "syr2k", "trmm"],
    "Linear\nAlgebra\nKernels": ["2mm", "3mm", "atax", "bicg", "doitgen", "mvt"],
    "Linear\nAlgebra\nSolvers": ["cholesky", "durbin", "gramschmidt", "lu", "ludcmp", "trisolv"],
    "Medley": ["deriche", "floyd-marshall", "nussinov"],
    "Stencils": ["adi", "fdtd-2d", "heat-3d", "jacobi-1d", "jacobi-2d", "seidel-2d"],
    "DP": ["floyd-warshall"]
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
                    print(f"Error processing line: {line}", file=sys.stderr)
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
    
    print("data[trmm]:", data[0])
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
        values['speedup'] = (values['llvm'] / values['mlir']) if values['mlir'] > 0.001 else 0.0
    
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

def prepare_plot_data_transposed(data):
    """
    Prepare data for plotting by organizing by category.
    For transposed version, categories are on x-axis, benchmarks on y-axis.
    Includes support for thicker bars, tighter within-category spacing, and gaps between categories.
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
    
    # Collect all data for plotting
    benchmarks_by_category = []
    speedups_by_category = []
    benchmark_positions_by_category = []
    category_positions = []
    category_benchmark_counts = []
    
    # Define spacing parameters
    within_category_spacing = 1.0  # Smaller value = tighter spacing within category
    between_category_spacing = 1.6  # Larger value = bigger gap between categories
    
    # Process data by category
    position = 0
    for category in categories:
        benchmarks = sorted(categorized_data[category].keys())
        if not benchmarks:
            continue
        
        benchmarks_by_category.append(benchmarks)
        
        # Collect speedups for this category
        category_speedups = [categorized_data[category][b]['speedup'] for b in benchmarks]
        speedups_by_category.append(category_speedups)
        
        # Calculate y-positions for benchmarks in this category with tighter spacing
        benchmark_positions = []
        for i in range(len(benchmarks)):
            benchmark_positions.append(position + (i * within_category_spacing))
        
        benchmark_positions_by_category.append(benchmark_positions)
        
        # Store category position
        category_positions.append(position + (len(benchmarks) * within_category_spacing / 2))
        category_benchmark_counts.append(len(benchmarks))
        
        # Move to next category position with a gap
        position += (len(benchmarks) * within_category_spacing) + between_category_spacing / 2.0
    
    # Flatten lists for plotting
    all_benchmarks = []
    all_positions = []
    all_speedups = []
    category_boundaries = []
    last_position = - between_category_spacing  # Start boundary before first category
    
    for i, (benchmarks, positions, speedups) in enumerate(zip(
            benchmarks_by_category, 
            benchmark_positions_by_category, 
            speedups_by_category)):
        
        # Add category boundary
        category_start = last_position + between_category_spacing/2
        category_boundaries.append(category_start)
        
        for j, (benchmark, position, speedup) in enumerate(zip(benchmarks, positions, speedups)):
            all_benchmarks.append(benchmark)
            all_positions.append(position)
            all_speedups.append(speedup)
            last_position = position
        
    # Add final boundary after the last category
    category_boundaries.append(last_position + between_category_spacing/2)

    
    return {
        'categorized_data': categorized_data,
        'all_benchmarks': all_benchmarks,
        'all_positions': all_positions,
        'all_speedups': all_speedups,
        'category_boundaries': category_boundaries,
        'categories': categories,
        'statistics': {
            'geo_mean_speedup': geo_mean_speedup,
            'mlir_wins': mlir_wins,
            'llvm_wins': llvm_wins,
            'total_benchmarks': len(data)
        }
    }

def plot_speedup_transposed(data, use_aot, binaryen_opt_level, output_file=None, title=None):
    """Plot speedup comparison chart with benchmarks on y-axis and categories grouped."""
    speedup_positive = '#009E73'  # Dark green
    speedup_negative = '#D55E00'  # Dark orange
    
    plot_data = prepare_plot_data_transposed(data)
    
    # Extract plotting data
    benchmarks = plot_data['all_benchmarks']
    positions = plot_data['all_positions']
    speedups = plot_data['all_speedups']
    categories = plot_data['categories']
    category_boundaries = plot_data['category_boundaries']
    
    # Set figure size based on number of benchmarks
    fig_height = max(10, len(benchmarks) * 0.4)
    fig = plt.figure(figsize=(12, fig_height))
    ax = fig.add_subplot(1, 1, 1)
    
    # Use custom title if provided, otherwise use default
    if title is None:
        execution_mode = "Interpreter" if not use_aot else "AOT"
        title = f'Speedup of WAMI over LLVM - {execution_mode}'
    
    # fig.suptitle(title, fontsize=30, y=1.00)
    
    # Plot horizontal bars with thicker height
    height = 0.8  # Increased bar thickness
    for pos, speedup, benchmark in zip(positions, speedups, benchmarks):
        color = speedup_positive if speedup > 1 else speedup_negative
        ax.barh(pos, speedup, height=height, color=color, alpha=0.85)
    
    # Reference line at speedup = 1
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
    
    # Determine x-axis limits
    min_speedup = min(speedups)
    max_speedup = max(speedups)
    padding = 0.05
    distance_below = 1.0 - min_speedup
    distance_above = max_speedup - 1.0
    max_distance = max(distance_below, distance_above)
    x_min = max(0, 1.0 - max_distance - padding)
    x_max = 1.0 + max_distance + padding
    
    if x_max - x_min < 0.5:
        x_min = max(0.7, 1.0 - 0.25)
        x_max = min(1.4, 1.0 + 0.25)
    
    ax.set_xlim(x_min, x_max)
    
    # X-ticks
    x_ticks = np.arange(
        np.floor(x_min * 10) / 10, 
        np.ceil(x_max * 10) / 10 + 0.01, 
        0.05
    )
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel('Speedup (LLVM/WAMI)', fontsize=15)
    
    # Y-ticks for benchmarks
    ax.set_yticks(positions)
    ax.set_yticklabels(benchmarks, fontsize=30)
    
    # Add horizontal lines and shaded areas between categories
    for i in range(len(category_boundaries) - 1):
        # Add horizontal separator lines at each boundary
        ax.axhline(y=category_boundaries[i], color='gray', linestyle='-', alpha=0.7, linewidth=1.5)
        
        # Add alternating background colors for categories
        if i % 2 == 0:
            ax.axhspan(category_boundaries[i], category_boundaries[i+1], 
                       facecolor='lightgray', alpha=0.2, zorder=-1)
    
    # Add final boundary line
    ax.axhline(y=category_boundaries[-1], color='gray', linestyle='-', alpha=0.7, linewidth=1.5)
    
    # Add category labels on the right side
    category_centers = []
    for i in range(len(categories)):
        # Calculate category center as midpoint between adjacent boundaries
        center = (category_boundaries[i] + category_boundaries[i+1]) / 2
        category_centers.append(center)
        
    # Create a twin y-axis for category labels
    # ax2 = ax.twinx()
    # ax2.set_ylim(ax.get_ylim())
    # ax2.set_yticks(category_centers)
    # ax2.set_yticklabels(categories, fontsize=24, fontweight='bold')
    # ax2.tick_params(axis='y', length=0)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=speedup_positive, alpha=0.85, label='WAMI faster'),
        Patch(facecolor=speedup_negative, alpha=0.85, label='LLVM faster')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.95, fontsize=24)
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Adjust margins
    # plt.subplots_adjust(left=0.2, right=0.99, bottom=0.03, top=0.95)
    plt.subplots_adjust(left=0.1, right=0.73, bottom=0.03, top=0.95)
    
    # Print summary stats
    stats = plot_data['statistics']
    summary_text = (
        f"Geometric Mean Speedup (LLVM/WAMI): {stats['geo_mean_speedup']:.3f}\n"
        f"Benchmarks where WAMI outperforms LLVM: {stats['mlir_wins']} out of {stats['total_benchmarks']}\n"
        f"Benchmarks where LLVM outperforms WAMI: {stats['llvm_wins']} out of {stats['total_benchmarks']}"
    )
    print("\nSummary Statistics:")
    print(summary_text)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        # Optional PDF
        if not output_file.endswith('.pdf'):
            pdf_file = f"{output_file.split('.')[0]}.pdf"
            plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
            print(f"PDF version saved to {pdf_file}")
    
    plt.show()

def plot_speedup_side_by_side(data, binaryen_opt_level, output_file=None, title=None):
    """Plot side-by-side speedup comparison charts for both interpreter and AOT modes."""
    speedup_positive = '#009E73'  # Dark green
    speedup_negative = '#D55E00'  # Dark orange
    
    # Extract raw data from the input data
    raw_data = parse_data_from_file(data) if isinstance(data, str) else data
    
    # Prepare data for both modes
    interpreter_data = filter_and_prepare_data(raw_data, False, binaryen_opt_level)
    aot_data = filter_and_prepare_data(raw_data, True, binaryen_opt_level)
    
    # Check if we have data for both modes
    if not interpreter_data and not aot_data:
        print("Error: No valid data found for either mode. Cannot create plots.")
        return
    elif not interpreter_data:
        print("Warning: Missing data for interpreter mode. Creating single plot for AOT mode.")
        plot_speedup_transposed(aot_data, True, binaryen_opt_level, output_file, title)
        return
    elif not aot_data:
        print("Warning: Missing data for AOT mode. Creating single plot for interpreter mode.")
        plot_speedup_transposed(interpreter_data, False, binaryen_opt_level, output_file, title)
        return
    
    # Process data for plotting
    interpreter_plot_data = prepare_plot_data_transposed(interpreter_data)
    aot_plot_data = prepare_plot_data_transposed(aot_data)
    
    # Set up figure and subplots
    fig_height = max(12, max(len(interpreter_plot_data['all_benchmarks']), 
                           len(aot_plot_data['all_benchmarks'])) * 0.4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, fig_height), sharey=True)
    
    # Use custom title if provided, otherwise use default
    if title is None:
        main_title = f'Speedup of WAMI over LLVM (Binaryen Opt Level {binaryen_opt_level})'
    else:
        main_title = title
    
    # fig.suptitle(main_title, fontsize=30, y=0.98)
    
    # Plot AOT mode (left subplot)
    ax1.set_title('AoT Compilation', fontsize=24)
    plot_subplot(ax1, aot_plot_data, speedup_positive, speedup_negative)

    # Plot interpreter mode (right subplot)
    ax2.set_title('Interpreter', fontsize=24)
    plot_subplot(ax2, interpreter_plot_data, speedup_positive, speedup_negative)
    
    # Set x-label for both subplots
    fig.text(0.5, 0.02, 'Speedup (LLVM/WAMI)', ha='center', fontsize=20)
    
    # Create a common legend for both subplots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=speedup_positive, alpha=0.85, label='WAMI faster'),
        Patch(facecolor=speedup_negative, alpha=0.85, label='LLVM faster')
    ]
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.9, 0.0),
               frameon=True, framealpha=0.95, fontsize=20, ncol=2)
    
    # Category labels for right subplot only (to avoid duplication)
    if 'categories' in aot_plot_data and 'category_boundaries' in aot_plot_data:
        categories = aot_plot_data['categories']
        category_boundaries = aot_plot_data['category_boundaries']
        
        if len(category_boundaries) > 1:
            category_centers = []
            for i in range(len(category_boundaries) - 1):
                center = (category_boundaries[i] + category_boundaries[i+1]) / 2
                category_centers.append(center)
            
            # Create a twin y-axis for category labels on right subplot only
            ax2_twin = ax2.twinx()
            ax2_twin.set_ylim(ax2.get_ylim())
            ax2_twin.set_yticks(category_centers)
            ax2_twin.set_yticklabels(categories, fontsize=20, fontweight='bold')
            ax2_twin.tick_params(axis='y', length=0)
    
    # Add summary statistics as text boxes
    int_stats = interpreter_plot_data['statistics']
    aot_stats = aot_plot_data['statistics']
    
    int_summary = (
        f"Geo Mean: {int_stats['geo_mean_speedup']:.3f}\n"
        f"WAMI wins: {int_stats['mlir_wins']}/{int_stats['total_benchmarks']}\n"
        f"LLVM wins: {int_stats['llvm_wins']}/{int_stats['total_benchmarks']}"
    )
    
    aot_summary = (
        f"Geo Mean: {aot_stats['geo_mean_speedup']:.3f}\n"
        f"WAMI wins: {aot_stats['mlir_wins']}/{aot_stats['total_benchmarks']}\n"
        f"LLVM wins: {aot_stats['llvm_wins']}/{aot_stats['total_benchmarks']}"
    )
    
    # ax1.text(0.02, 0.02, int_summary, transform=ax1.transAxes, 
    #          fontsize=20, bbox=dict(facecolor='white', alpha=0.8))
    
    # ax2.text(0.02, 0.02, aot_summary, transform=ax2.transAxes, 
    #          fontsize=20, bbox=dict(facecolor='white', alpha=0.8))
    
    # Print summary statistics to console
    print("\nInterpreter Mode Summary Statistics:")
    print(int_summary.replace('\n', ', '))
    
    print("\nAOT Mode Summary Statistics:")
    print(aot_summary.replace('\n', ', '))
    
    # Adjust layout
    plt.tight_layout(rect=[0.027, 0.05, 0.95, 0.99])
    
    # Save plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        
        # Optional PDF
        if not output_file.endswith('.pdf'):
            pdf_file = f"{output_file.split('.')[0]}.pdf"
            plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
            print(f"PDF version saved to {pdf_file}")
    
    plt.show()

def plot_subplot(ax, plot_data, speedup_positive, speedup_negative):
    """Plot a single subplot with the given data."""
    # Extract plotting data
    benchmarks = plot_data['all_benchmarks']
    positions = plot_data['all_positions']
    speedups = plot_data['all_speedups']
    category_boundaries = plot_data['category_boundaries']
    categories = plot_data['categories']
    
    # Plot horizontal bars with thicker height
    height = 0.8  # Increased bar thickness
    for pos, speedup, benchmark in zip(positions, speedups, benchmarks):
        color = speedup_positive if speedup > 1 else speedup_negative
        ax.barh(pos, speedup, height=height, color=color, alpha=0.85)
    
    # Reference line at speedup = 1
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
    
    # Determine x-axis limits
    min_speedup = min(speedups) if speedups else 0.5
    max_speedup = max(speedups) if speedups else 1.5
    padding = 0.05
    distance_below = 1.0 - min_speedup
    distance_above = max_speedup - 1.0
    max_distance = max(distance_below, distance_above)
    x_min = max(0, 1.0 - max_distance - padding)
    x_max = 1.0 + max_distance + padding
    
    if x_max - x_min < 0.5:
        x_min = max(0.7, 1.0 - 0.25)
        x_max = min(1.4, 1.0 + 0.25)
    
    ax.set_xlim(x_min, x_max)
    
    # X-ticks with increased spacing for side-by-side plots
    x_ticks = np.arange(
        np.floor(x_min * 10) / 10, 
        np.ceil(x_max * 10) / 10 + 0.01, 
        0.1
    )
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='x', labelsize=12)
    
    # Y-ticks for benchmarks
    ax.set_yticks(positions)
    ax.set_yticklabels(benchmarks, fontsize=20)
    
    # Add horizontal lines and shaded areas between categories
    for i in range(len(category_boundaries) - 1):
        # Add horizontal separator lines at each boundary
        ax.axhline(y=category_boundaries[i], color='gray', linestyle='-', alpha=0.7, linewidth=1.5)
        
        # Add alternating background colors for categories
        if i % 2 == 0:
            ax.axhspan(category_boundaries[i], category_boundaries[i+1], 
                       facecolor='lightgray', alpha=0.2, zorder=-1)
    
    # Add final boundary line
    if category_boundaries:
        ax.axhline(y=category_boundaries[-1], color='gray', linestyle='-', alpha=0.7, linewidth=1.5)
    
    # Calculate category centers for labels (only for right subplot)
    category_centers = []
    if len(category_boundaries) > 1:
        for i in range(len(category_boundaries) - 1):
            center = (category_boundaries[i] + category_boundaries[i+1]) / 2
            category_centers.append(center)
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.3)

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
    parser.add_argument('--title', help='Custom title for the plot (optional)')
    parser.add_argument('--transpose', action='store_true', default=True,
                        help='Generate a transposed plot with benchmarks on y-axis (default: True)')
    
    args = parser.parse_args()
    
    print("args.filename", args.filename)
    data = parse_data_from_file(args.filename)
    print("data", data)
    
    # Use transposed plot by default
    plot_speedup_side_by_side(data, args.binaryen_opt_level, 
                  output_file=args.output, title=args.title)

if __name__ == "__main__":
    main()