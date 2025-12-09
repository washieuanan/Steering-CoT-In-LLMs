"""
Evaluation Module for Reasoning Vector Intervention Experiments

This module provides tools for analyzing Phase B causal intervention results,
including data loading, metric computation, visualization, and statistical testing.

Usage:
    from evaluation import load_data, metrics, plots, statistical_tests
    
    # Load data from results directory
    df = load_data.load_all_runs('vm_results')
    
    # Compute metrics
    summary = metrics.compute_summary(df)
    
    # Create plots
    plots.plot_alpha_sweep(df)
    
    # Run statistical tests
    results = statistical_tests.run_all_tests(df)
"""

from . import load_data
from . import metrics
from . import plots
from . import statistical_tests

__version__ = "0.1.0"
__all__ = ["load_data", "metrics", "plots", "statistical_tests"]
