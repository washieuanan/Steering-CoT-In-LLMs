#!/usr/bin/env python3
"""
CLI Script for Running Intervention Analysis

Usage:
    python evaluation/run_analysis.py --results-dir vm_results --output-dir evaluation/outputs
    python evaluation/run_analysis.py --results-dir vm_results --output-dir evaluation/outputs --mode add
    python evaluation/run_analysis.py --help
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import load_data, metrics, plots, statistical_tests


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run analysis on Phase B intervention experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python evaluation/run_analysis.py --results-dir vm_results
    
    # Full analysis with plots
    python evaluation/run_analysis.py --results-dir vm_results --output-dir evaluation/outputs --plots
    
    # Filter to specific mode
    python evaluation/run_analysis.py --results-dir vm_results --mode add
    
    # Filter to specific locality
    python evaluation/run_analysis.py --results-dir vm_results --locality cot
        """
    )
    
    parser.add_argument(
        '--results-dir', '-r',
        type=str,
        default='vm_results',
        help='Path to results directory (default: vm_results)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Directory to save outputs (default: print to stdout)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['add', 'random', 'lesion', 'rescue'],
        default=None,
        help='Filter to specific intervention mode'
    )
    
    parser.add_argument(
        '--locality', '-l',
        type=str,
        choices=['cot', 'answer', 'full'],
        default=None,
        help='Filter to specific locality'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        choices=['answer', 'reasoning'],
        default='answer',
        help='Metric to analyze (default: answer)'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate and save plots'
    )
    
    parser.add_argument(
        '--statistical-tests',
        action='store_true',
        help='Run statistical tests'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all analyses (plots, statistical tests, etc.)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['pdf', 'png', 'svg'],
        default='pdf',
        help='Output format for plots (default: pdf)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Load data
    if not args.quiet:
        print(f"\n{'='*60}")
        print("Loading data...")
        print(f"{'='*60}\n")
    
    try:
        df = load_data.load_all_runs(args.results_dir)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Print data summary
    if not args.quiet:
        load_data.print_data_summary(df)
    
    # Apply filters
    if args.mode:
        df = load_data.filter_by_mode(df, [args.mode])
        if not args.quiet:
            print(f"\nFiltered to mode: {args.mode}")
    
    if args.locality:
        df = load_data.filter_by_locality(df, args.locality)
        if not args.quiet:
            print(f"Filtered to locality: {args.locality}")
    
    if len(df) == 0:
        print("Error: No data after filtering")
        sys.exit(1)
    
    # Compute summary statistics
    if not args.quiet:
        print(f"\n{'='*60}")
        print("Summary Statistics")
        print(f"{'='*60}\n")
    
    summary = metrics.compute_summary(df)
    
    if not args.quiet:
        for key, value in summary.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    # Aggregate by configuration
    if not args.quiet:
        print(f"\n{'='*60}")
        print("Aggregated Results by Configuration")
        print(f"{'='*60}\n")
    
    agg_df = metrics.aggregate_by_config(df, ['mode', 'layer', 'alpha'], args.metric)
    
    if not args.quiet and len(agg_df) > 0:
        print(agg_df.to_string())
    
    # Save aggregated results
    if output_dir:
        agg_df.to_csv(output_dir / 'aggregated_results.csv', index=False)
        
        # Save summary as JSON
        summary_serializable = {k: (float(v) if isinstance(v, (int, float)) and v is not None else v) 
                                for k, v in summary.items()}
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary_serializable, f, indent=2, default=str)
    
    # Run statistical tests
    if args.statistical_tests or args.all:
        if not args.quiet:
            print(f"\n{'='*60}")
            print("Statistical Tests")
            print(f"{'='*60}")
        
        test_results = statistical_tests.run_all_tests(df, args.metric)
        
        if not args.quiet:
            statistical_tests.print_test_summary(test_results)
        
        # Save test results
        if output_dir:
            for test_name, test_df in test_results.items():
                if len(test_df) > 0:
                    test_df.to_csv(output_dir / f'test_{test_name}.csv', index=False)
    
    # Generate plots
    if args.plots or args.all:
        if output_dir is None:
            print("Warning: --output-dir required for saving plots")
        else:
            if not args.quiet:
                print(f"\n{'='*60}")
                print("Generating Plots")
                print(f"{'='*60}\n")
            
            plots.save_all_plots(df, output_dir, args.metric, args.format)
    
    # Final summary
    if not args.quiet:
        print(f"\n{'='*60}")
        print("Analysis Complete")
        print(f"{'='*60}")
        
        if output_dir:
            print(f"\nResults saved to: {output_dir}")
            print(f"\nFiles generated:")
            for f in sorted(output_dir.glob('*')):
                print(f"  - {f.name}")
        
        # Key findings
        print(f"\nKey Findings:")
        
        # Delta accuracy by mode
        if 'mode' in df.columns:
            for mode in df['mode'].unique():
                mode_df = df[df['mode'] == mode]
                delta = metrics.compute_delta_accuracy(mode_df, args.metric)
                print(f"  {mode.capitalize()} mode: Δ{args.metric} = {delta:+.4f}")
        
        # Specificity
        specificity = metrics.compute_specificity(df, args.metric)
        if not pd.isna(specificity):
            print(f"  Specificity (add - random): {specificity:+.4f}")
        
        print("")


if __name__ == '__main__':
    import pandas as pd  # Import for isna check
    main()
