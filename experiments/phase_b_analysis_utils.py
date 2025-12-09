"""
Phase B Analysis Utilities

This module provides functions for analyzing Phase B causal intervention results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, ttest_1samp, binom
from pathlib import Path
import gzip
import json
import os
from typing import Dict, List, Tuple, Optional

# Color palette
MODE_COLORS = {'add': '#2ecc71', 'random': '#e74c3c', 'lesion': '#9b59b6'}
MODEL_COLORS = {
    'Llama-3.1-8B-Instruct': '#3498db',
    'Mistral-7B-Instruct-v0.3': '#e67e22',
    'Qwen2.5-7B-Instruct': '#1abc9c'
}
DATASET_COLORS = {'arc': '#3498db', 'gsm8k': '#e74c3c', 'mmlu_pro': '#9b59b6'}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def aggregate_grid_csvs(phase_b_root: Path) -> pd.DataFrame:
    """Aggregate all grid.csv files from Phase B results."""
    all_rows = []
    for run_dir in phase_b_root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue
        parts = run_dir.name.split('__')
        if len(parts) < 2:
            continue
        model_name = parts[0]
        timestamp = parts[1].replace('intv_', '') if len(parts) > 1 else 'unknown'
        for dataset_dir in run_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            grid_path = dataset_dir / 'grid.csv'
            if not grid_path.exists():
                continue
            try:
                df = pd.read_csv(grid_path)
                df['model'] = model_name
                df['dataset'] = dataset_name
                df['timestamp'] = timestamp
                df['run_dir'] = str(run_dir)
                all_rows.append(df)
            except Exception as e:
                print(f'Error reading {grid_path}: {e}')
    if not all_rows:
        raise ValueError('No grid.csv files found')
    return pd.concat(all_rows, ignore_index=True)


def aggregate_ara_csvs(phase_b_root: Path) -> pd.DataFrame:
    """Aggregate all ara_summary.csv files from Phase B results."""
    all_rows = []
    for run_dir in phase_b_root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue
        parts = run_dir.name.split('__')
        if len(parts) < 2:
            continue
        model_name = parts[0]
        timestamp = parts[1].replace('intv_', '') if len(parts) > 1 else 'unknown'
        for dataset_dir in run_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            ara_path = dataset_dir / 'ara_summary.csv'
            if not ara_path.exists():
                continue
            try:
                df = pd.read_csv(ara_path)
                df['model'] = model_name
                df['dataset'] = dataset_name
                df['timestamp'] = timestamp
                df['run_dir'] = str(run_dir)
                all_rows.append(df)
            except Exception as e:
                print(f'Error reading {ara_path}: {e}')
    if not all_rows:
        return pd.DataFrame()  # Return empty if no ARA data
    return pd.concat(all_rows, ignore_index=True)


def aggregate_paired_csvs(phase_b_root: Path) -> pd.DataFrame:
    """Aggregate all paired_*.csv.gz files from Phase B results (for runs without grid.csv)."""
    all_rows = []
    for run_dir in phase_b_root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue
        parts = run_dir.name.split('__')
        if len(parts) < 2:
            continue
        model_name = parts[0]
        timestamp = parts[1].replace('intv_', '') if len(parts) > 1 else 'unknown'
        for dataset_dir in run_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            # Skip if grid.csv exists (already processed)
            if (dataset_dir / 'grid.csv').exists():
                continue
            runs_dir = dataset_dir / 'runs'
            if not runs_dir.exists():
                continue
            for csv_file in runs_dir.glob('paired_*.csv.gz'):
                try:
                    with gzip.open(csv_file, 'rt', encoding='utf-8') as f:
                        df = pd.read_csv(f)
                    df['model'] = model_name
                    df['dataset'] = dataset_name
                    df['timestamp'] = timestamp
                    df['run_dir'] = str(run_dir)
                    df['source_file'] = csv_file.name
                    all_rows.append(df)
                except Exception as e:
                    print(f'Error reading {csv_file}: {e}')
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def load_paired_csv(csv_path: Path) -> pd.DataFrame:
    """Load a paired generation CSV (may be gzipped)."""
    if csv_path.suffix == '.gz':
        with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
            return pd.read_csv(f)
    return pd.read_csv(csv_path)


def get_latest_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only most recent run for each configuration."""
    df_sorted = df.sort_values('timestamp', ascending=False)
    df_dedup = df_sorted.drop_duplicates(
        subset=['model', 'dataset', 'mode', 'layer', 'alpha'], keep='first'
    )
    return df_dedup.sort_values(['model', 'dataset', 'mode', 'alpha']).reset_index(drop=True)


def get_latest_ara_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only most recent ARA run for each configuration."""
    if len(df) == 0:
        return df
    df_sorted = df.sort_values('timestamp', ascending=False)
    df_dedup = df_sorted.drop_duplicates(
        subset=['model', 'dataset', 'layer'], keep='first'
    )
    return df_dedup.sort_values(['model', 'dataset']).reset_index(drop=True)


def load_phase_b_data(phase_b_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and deduplicate Phase B data.
    
    Returns:
        Tuple of (grid_df, ara_df)
    """
    grid_df = aggregate_grid_csvs(phase_b_root)
    grid_df = get_latest_runs(grid_df)
    
    ara_df = aggregate_ara_csvs(phase_b_root)
    ara_df = get_latest_ara_runs(ara_df)
    
    return grid_df, ara_df


def load_phase_b_grid_only(phase_b_root: Path) -> pd.DataFrame:
    """Load only grid.csv data (for backward compatibility)."""
    df_all = aggregate_grid_csvs(phase_b_root)
    return get_latest_runs(df_all)


# ==============================================================================
# ARA (ADD-REMOVE-ADD) ANALYSIS FUNCTIONS
# ==============================================================================

def plot_ara_consistency(ara_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot ARA consistency rates by model and dataset."""
    if len(ara_df) == 0:
        print('No ARA data available')
        return None, None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Answer consistency
    ax = axes[0]
    pivot = ara_df.pivot_table(values='consistency_rate_answer', 
                                index='dataset', columns='model')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax, vmin=0, vmax=1)
    ax.set_title('ARA: Answer Consistency Rate\n(P(Pass1 == Pass3))')
    
    # Reasoning consistency
    ax = axes[1]
    if 'consistency_rate_reasoning' in ara_df.columns:
        pivot = ara_df.pivot_table(values='consistency_rate_reasoning', 
                                    index='dataset', columns='model')
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax, vmin=0, vmax=1)
        ax.set_title('ARA: Reasoning Consistency Rate\n(P(Pass1 == Pass3))')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_ara_recovery(ara_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot ARA recovery rates."""
    if len(ara_df) == 0:
        print('No ARA data available')
        return None, None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'recovery_rate_answer' in ara_df.columns:
        pivot = ara_df.pivot_table(values='recovery_rate_answer', 
                                    index='dataset', columns='model')
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, vmin=0, vmax=1)
        ax.set_title('ARA: Recovery Rate\n(P(Pass1 != Pass2) - Intervention had effect)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_ara_accuracy_progression(ara_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot accuracy across the three ARA passes."""
    if len(ara_df) == 0:
        print('No ARA data available')
        return None, None
    
    # Reshape for plotting
    metrics = []
    for _, row in ara_df.iterrows():
        model = row['model']
        dataset = row['dataset']
        for pass_num in [1, 2, 3]:
            metrics.append({
                'model': model,
                'dataset': dataset,
                'pass': f'Pass {pass_num}',
                'answer_accuracy': row[f'pass{pass_num}_acc_answer'],
                'reasoning_accuracy': row.get(f'pass{pass_num}_acc_reasoning', np.nan)
            })
    
    metrics_df = pd.DataFrame(metrics)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group bar chart
    x = np.arange(len(ara_df['model'].unique()) * len(ara_df['dataset'].unique()))
    width = 0.25
    
    for i, pass_name in enumerate(['Pass 1', 'Pass 2', 'Pass 3']):
        pass_data = metrics_df[metrics_df['pass'] == pass_name]
        pass_data = pass_data.groupby(['model', 'dataset'])['answer_accuracy'].mean().reset_index()
        labels = [f"{r['model'][:8]}\n{r['dataset']}" for _, r in pass_data.iterrows()]
        values = pass_data['answer_accuracy'].values
        ax.bar(np.arange(len(values)) + i * width, values, width, label=pass_name)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('ARA: Accuracy Across Passes\n(Pass1: Add, Pass2: Remove, Pass3: Add again)')
    ax.set_xticks(np.arange(len(labels)) + width)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def create_ara_summary_table(ara_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary table for ARA results."""
    if len(ara_df) == 0:
        return pd.DataFrame()
    
    summary = ara_df[['model', 'dataset', 'n', 
                       'pass1_acc_answer', 'pass2_acc_answer', 'pass3_acc_answer',
                       'consistency_rate_answer', 'recovery_rate_answer']].copy()
    summary.columns = ['Model', 'Dataset', 'N', 
                       'Pass1 Acc', 'Pass2 Acc', 'Pass3 Acc',
                       'Consistency', 'Recovery']
    return summary.round(3)


def run_ara_statistical_tests(ara_df: pd.DataFrame) -> Dict:
    """Run statistical tests on ARA data."""
    results = {}
    
    if len(ara_df) == 0:
        return results
    
    # Test: Is consistency rate significantly above 0.5?
    # (If intervention has no effect, P(P1==P3) should be high due to randomness)
    consistency_rates = ara_df['consistency_rate_answer'].dropna()
    if len(consistency_rates) > 1:
        t_stat, p_value = ttest_1samp(consistency_rates, 0.5, alternative='greater')
        results['consistency_vs_chance'] = {
            'mean': consistency_rates.mean(),
            'std': consistency_rates.std(),
            't_stat': t_stat,
            'p_value': p_value,
            'interpretation': 'High consistency suggests reproducible intervention effect'
        }
    
    # Test: Is recovery rate significantly different from 0?
    # Recovery = P(P1 != P2) - how often does removing the intervention change output
    if 'recovery_rate_answer' in ara_df.columns:
        recovery_rates = ara_df['recovery_rate_answer'].dropna()
        if len(recovery_rates) > 1:
            t_stat, p_value = ttest_1samp(recovery_rates, 0, alternative='greater')
            results['recovery_vs_zero'] = {
                'mean': recovery_rates.mean(),
                'std': recovery_rates.std(),
                't_stat': t_stat,
                'p_value': p_value,
                'interpretation': 'Low recovery suggests intervention does not reliably change output'
            }
    
    return results


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_alpha_sweep_overall(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot overall alpha sweep comparing add vs random modes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in ['add', 'random']:
        mode_df = df[df['mode'] == mode]
        if len(mode_df) == 0:
            continue
        grouped = mode_df.groupby('alpha')['delta_answer'].agg(['mean', 'std', 'count'])
        grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
        color = MODE_COLORS.get(mode, 'gray')
        ax.plot(grouped.index, grouped['mean'], 'o-', label=mode.capitalize(),
                color=color, linewidth=2, markersize=8)
        ax.fill_between(grouped.index,
                        grouped['mean'] - 1.96 * grouped['sem'],
                        grouped['mean'] + 1.96 * grouped['sem'],
                        alpha=0.2, color=color)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Alpha (Intervention Strength)')
    ax.set_ylabel('Δ Answer Accuracy')
    ax.set_title('Effect of Intervention Strength on Answer Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_alpha_sweep_by_dataset(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot alpha sweep faceted by dataset."""
    datasets = df['dataset'].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        dataset_df = df[df['dataset'] == dataset]
        for mode in ['add', 'random']:
            mode_df = dataset_df[dataset_df['mode'] == mode]
            if len(mode_df) == 0:
                continue
            grouped = mode_df.groupby('alpha')['delta_answer'].agg(['mean', 'std', 'count'])
            grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
            color = MODE_COLORS.get(mode, 'gray')
            ax.plot(grouped.index, grouped['mean'], 'o-', label=mode, color=color, linewidth=2)
            ax.fill_between(grouped.index,
                            grouped['mean'] - 1.96 * grouped['sem'],
                            grouped['mean'] + 1.96 * grouped['sem'],
                            alpha=0.2, color=color)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Alpha')
        ax.set_title(dataset.upper())
        ax.legend()
    axes[0].set_ylabel('Δ Answer Accuracy')
    plt.suptitle('Alpha Sweep by Dataset', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_alpha_sweep_by_model(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot alpha sweep faceted by model."""
    models = df['model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        model_df = df[df['model'] == model]
        for mode in ['add', 'random']:
            mode_df = model_df[model_df['mode'] == mode]
            if len(mode_df) == 0:
                continue
            grouped = mode_df.groupby('alpha')['delta_answer'].agg(['mean', 'std', 'count'])
            grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
            color = MODE_COLORS.get(mode, 'gray')
            ax.plot(grouped.index, grouped['mean'], 'o-', label=mode, color=color, linewidth=2)
            ax.fill_between(grouped.index,
                            grouped['mean'] - 1.96 * grouped['sem'],
                            grouped['mean'] + 1.96 * grouped['sem'],
                            alpha=0.2, color=color)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Alpha')
        ax.set_title(model[:25])
        ax.legend()
    axes[0].set_ylabel('Δ Answer Accuracy')
    plt.suptitle('Alpha Sweep by Model', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_heatmap_delta(df: pd.DataFrame, mode: str = 'add', alpha_filter: str = 'positive',
                       save_path: Optional[Path] = None):
    """Plot heatmap of delta accuracy by dataset and model."""
    if alpha_filter == 'positive':
        filtered = df[(df['mode'] == mode) & (df['alpha'] > 0)]
        title_suffix = 'α > 0'
    elif alpha_filter == 'negative':
        filtered = df[(df['mode'] == mode) & (df['alpha'] < 0)]
        title_suffix = 'α < 0'
    else:
        filtered = df[df['mode'] == mode]
        title_suffix = 'all α'

    if len(filtered) == 0:
        print(f'No data for {mode} mode with {alpha_filter} alpha')
        return None, None

    hm_data = filtered.groupby(['dataset', 'model'])['delta_answer'].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(hm_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title(f'Mean Δ Answer Accuracy ({mode.capitalize()} Mode, {title_suffix})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_heatmap_specificity(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot heatmap of add - random effect difference."""
    add_df = df[df['mode'] == 'add'].groupby(['dataset', 'model', 'alpha'])['delta_answer'].mean().reset_index()
    random_df = df[df['mode'] == 'random'].groupby(['dataset', 'model', 'alpha'])['delta_answer'].mean().reset_index()
    merged = add_df.merge(random_df, on=['dataset', 'model', 'alpha'], suffixes=('_add', '_random'))
    
    if len(merged) == 0:
        print('Insufficient data for add vs random comparison')
        return None, None
    
    merged['delta_diff'] = merged['delta_answer_add'] - merged['delta_answer_random']
    hm_data = merged.groupby(['dataset', 'model'])['delta_diff'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(hm_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Specificity: Add Effect - Random Effect\n(Positive = Add is better)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_flip_bars(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot stacked bar chart of flips per alpha."""
    if 'answer_wrong_to_right' not in df.columns:
        print('Flip columns not available')
        return None, None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mode in zip(axes, ['add', 'random']):
        mode_df = df[df['mode'] == mode]
        if len(mode_df) == 0:
            continue
        grouped = mode_df.groupby('alpha')[['answer_wrong_to_right', 'answer_right_to_wrong']].sum()
        grouped.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Count')
        ax.set_title(f'{mode.capitalize()} Mode: Flips')
        ax.legend(['Wrong→Right', 'Right→Wrong'])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, axes


def plot_net_gain_line(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot net gain (wrong→right - right→wrong) line plot."""
    if 'answer_net_gain' not in df.columns:
        print('Net gain column not available')
        return None, None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in ['add', 'random']:
        mode_df = df[df['mode'] == mode]
        if len(mode_df) == 0:
            continue
        grouped = mode_df.groupby('alpha')['answer_net_gain'].agg(['mean', 'std', 'count'])
        grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
        color = MODE_COLORS.get(mode, 'gray')
        ax.plot(grouped.index, grouped['mean'], 'o-', label=mode, color=color, linewidth=2)
        ax.fill_between(grouped.index,
                        grouped['mean'] - 1.96 * grouped['sem'],
                        grouped['mean'] + 1.96 * grouped['sem'],
                        alpha=0.2, color=color)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Net Gain (Wrong→Right - Right→Wrong)')
    ax.set_title('Net Benefit of Intervention')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


# ==============================================================================
# STATISTICAL TESTS
# ==============================================================================

def mcnemar_test(wrong_to_right: int, right_to_wrong: int) -> float:
    """Run exact McNemar's test using binomial distribution."""
    n = wrong_to_right + right_to_wrong
    if n == 0:
        return 1.0
    k = min(wrong_to_right, right_to_wrong)
    # Two-sided exact McNemar's test
    p_value = 2 * binom.cdf(k, n, 0.5)
    return min(p_value, 1.0)


def run_mcnemar_tests(df: pd.DataFrame, mode: str = 'add',
                      alpha_filter: Optional[float] = None) -> pd.DataFrame:
    """Run McNemar's test for each model-dataset combination."""
    results = []
    mode_df = df[df['mode'] == mode]
    
    if alpha_filter is not None:
        if alpha_filter >= 0:
            mode_df = mode_df[mode_df['alpha'] > 0]
        else:
            mode_df = mode_df[mode_df['alpha'] < 0]
    
    for (model, dataset), group in mode_df.groupby(['model', 'dataset']):
        w2r = int(group['answer_wrong_to_right'].sum())
        r2w = int(group['answer_right_to_wrong'].sum())
        p_value = mcnemar_test(w2r, r2w)
        
        results.append({
            'model': model,
            'dataset': dataset,
            'mode': mode,
            'wrong_to_right': w2r,
            'right_to_wrong': r2w,
            'net_gain': w2r - r2w,
            'p_value': p_value,
            'significant_0.05': p_value < 0.05,
            'significant_0.01': p_value < 0.01,
        })
    
    return pd.DataFrame(results)


def run_sign_test(df: pd.DataFrame, mode: str = 'add',
                  alpha_filter: Optional[float] = None) -> Dict:
    """Run sign test on delta values."""
    mode_df = df[df['mode'] == mode].copy()
    
    if alpha_filter is not None:
        if alpha_filter >= 0:
            mode_df = mode_df[mode_df['alpha'] > 0]
        else:
            mode_df = mode_df[mode_df['alpha'] < 0]
    
    deltas = mode_df['delta_answer'].dropna()
    n_positive = (deltas > 0).sum()
    n_negative = (deltas < 0).sum()
    n_total = n_positive + n_negative
    
    if n_total == 0:
        return {'n_positive': 0, 'n_negative': 0, 'p_value': 1.0}
    
    # Two-sided sign test
    p_value = 2 * binom.cdf(min(n_positive, n_negative), n_total, 0.5)
    p_value = min(p_value, 1.0)
    
    return {
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_zero': len(deltas) - n_total,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def run_ttest_vs_zero(df: pd.DataFrame, mode: str = 'add',
                      alpha_filter: Optional[float] = None,
                      alternative: str = 'two-sided') -> Dict:
    """Run one-sample t-test against zero."""
    mode_df = df[df['mode'] == mode].copy()
    
    if alpha_filter is not None:
        if alpha_filter >= 0:
            mode_df = mode_df[mode_df['alpha'] > 0]
        else:
            mode_df = mode_df[mode_df['alpha'] < 0]
    
    deltas = mode_df['delta_answer'].dropna()
    
    if len(deltas) < 2:
        return {'mean': np.nan, 't_stat': np.nan, 'p_value': 1.0}
    
    t_stat, p_value = ttest_1samp(deltas, 0, alternative=alternative)
    
    return {
        'mean': deltas.mean(),
        'std': deltas.std(),
        'n': len(deltas),
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def run_mann_whitney(df: pd.DataFrame, alpha_value: Optional[float] = None) -> Dict:
    """Run Mann-Whitney U test comparing add vs random modes."""
    add_df = df[df['mode'] == 'add']
    random_df = df[df['mode'] == 'random']
    
    if alpha_value is not None:
        add_df = add_df[add_df['alpha'] == alpha_value]
        random_df = random_df[random_df['alpha'] == alpha_value]
    
    add_deltas = add_df['delta_answer'].dropna()
    random_deltas = random_df['delta_answer'].dropna()
    
    if len(add_deltas) < 2 or len(random_deltas) < 2:
        return {'U_stat': np.nan, 'p_value': 1.0}
    
    U_stat, p_value = mannwhitneyu(add_deltas, random_deltas, alternative='two-sided')
    
    # Effect size (rank-biserial correlation)
    n1, n2 = len(add_deltas), len(random_deltas)
    r = 1 - (2 * U_stat) / (n1 * n2)
    
    return {
        'add_mean': add_deltas.mean(),
        'random_mean': random_deltas.mean(),
        'add_n': n1,
        'random_n': n2,
        'U_stat': U_stat,
        'p_value': p_value,
        'effect_size_r': r,
        'significant': p_value < 0.05
    }


def run_wilcoxon_test(df: pd.DataFrame, mode: str = 'add',
                      alpha_filter: Optional[float] = None) -> Dict:
    """Run Wilcoxon signed-rank test on delta values."""
    mode_df = df[df['mode'] == mode].copy()
    
    if alpha_filter is not None:
        if alpha_filter >= 0:
            mode_df = mode_df[mode_df['alpha'] > 0]
        else:
            mode_df = mode_df[mode_df['alpha'] < 0]
    
    deltas = mode_df['delta_answer'].dropna()
    deltas = deltas[deltas != 0]  # Remove zeros
    
    if len(deltas) < 2:
        return {'W_stat': np.nan, 'p_value': 1.0}
    
    try:
        W_stat, p_value = wilcoxon(deltas, alternative='two-sided')
    except ValueError:
        return {'W_stat': np.nan, 'p_value': 1.0}
    
    return {
        'median': deltas.median(),
        'n': len(deltas),
        'W_stat': W_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def compute_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cohen's d for add vs random per dataset and model."""
    results = []
    
    for (dataset, model), group in df.groupby(['dataset', 'model']):
        add_deltas = group[group['mode'] == 'add']['delta_answer'].dropna().values
        random_deltas = group[group['mode'] == 'random']['delta_answer'].dropna().values
        
        d = cohens_d(add_deltas, random_deltas)
        
        # Interpret effect size
        if np.isnan(d):
            interpretation = 'N/A'
        elif abs(d) < 0.2:
            interpretation = 'negligible'
        elif abs(d) < 0.5:
            interpretation = 'small'
        elif abs(d) < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        results.append({
            'dataset': dataset,
            'model': model,
            'add_mean': add_deltas.mean() if len(add_deltas) > 0 else np.nan,
            'random_mean': random_deltas.mean() if len(random_deltas) > 0 else np.nan,
            'cohens_d': d,
            'interpretation': interpretation
        })
    
    return pd.DataFrame(results)


def run_dose_response_regression(df: pd.DataFrame, mode: str = 'add') -> Dict:
    """Fit linear regression of delta ~ alpha."""
    mode_df = df[df['mode'] == mode].copy()
    
    alphas = mode_df['alpha'].values
    deltas = mode_df['delta_answer'].values
    
    # Remove NaN
    mask = ~np.isnan(deltas)
    alphas = alphas[mask]
    deltas = deltas[mask]
    
    if len(alphas) < 3:
        return {'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan, 'p_value': 1.0}
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(alphas, deltas)
    
    return {
        'slope': slope,
        'slope_se': std_err,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


# ==============================================================================
# SUMMARY FUNCTIONS
# ==============================================================================

def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary table grouped by model, dataset, mode."""
    agg_dict = {
        'n': 'mean',
        'acc_base_answer': 'mean',
        'acc_intv_answer': 'mean',
        'delta_answer': ['mean', 'std', 'count']
    }
    
    if 'answer_net_gain' in df.columns:
        agg_dict['answer_net_gain'] = 'sum'
    
    summary = df.groupby(['model', 'dataset', 'mode']).agg(agg_dict).round(3)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()


def run_all_statistical_tests(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Run all statistical tests and return results as dictionary of DataFrames."""
    results = {}
    
    # McNemar tests
    mcnemar_positive = run_mcnemar_tests(df, mode='add', alpha_filter=0)
    mcnemar_negative = run_mcnemar_tests(df, mode='add', alpha_filter=-1)
    results['mcnemar_add_positive'] = mcnemar_positive
    results['mcnemar_add_negative'] = mcnemar_negative
    
    # Sign tests
    sign_add_pos = run_sign_test(df, mode='add', alpha_filter=0)
    sign_add_neg = run_sign_test(df, mode='add', alpha_filter=-1)
    results['sign_test_summary'] = pd.DataFrame([
        {'condition': 'add_positive', **sign_add_pos},
        {'condition': 'add_negative', **sign_add_neg}
    ])
    
    # T-tests
    ttest_add_pos = run_ttest_vs_zero(df, mode='add', alpha_filter=0)
    ttest_add_neg = run_ttest_vs_zero(df, mode='add', alpha_filter=-1, alternative='less')
    results['ttest_summary'] = pd.DataFrame([
        {'condition': 'add_positive_two_sided', **ttest_add_pos},
        {'condition': 'add_negative_one_sided', **ttest_add_neg}
    ])
    
    # Mann-Whitney for each alpha
    mw_results = []
    for alpha in df['alpha'].unique():
        mw = run_mann_whitney(df, alpha_value=alpha)
        mw['alpha'] = alpha
        mw_results.append(mw)
    results['mann_whitney'] = pd.DataFrame(mw_results)
    
    # Effect sizes
    results['effect_sizes'] = compute_effect_sizes(df)
    
    # Dose-response
    dr_add = run_dose_response_regression(df, mode='add')
    dr_random = run_dose_response_regression(df, mode='random')
    results['dose_response'] = pd.DataFrame([
        {'mode': 'add', **dr_add},
        {'mode': 'random', **dr_random}
    ])
    
    return results


def print_results_summary(test_results: Dict[str, pd.DataFrame]):
    """Print formatted summary of statistical test results."""
    print("\n" + "=" * 80)
    print("STATISTICAL TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n--- McNemar's Test (Add Mode, α > 0) ---")
    print(test_results['mcnemar_add_positive'].to_string())
    
    print("\n--- McNemar's Test (Add Mode, α < 0) ---")
    print(test_results['mcnemar_add_negative'].to_string())
    
    print("\n--- Sign Test ---")
    print(test_results['sign_test_summary'].to_string())
    
    print("\n--- One-Sample t-test vs Zero ---")
    print(test_results['ttest_summary'].to_string())
    
    print("\n--- Mann-Whitney U (Add vs Random) ---")
    print(test_results['mann_whitney'].to_string())
    
    print("\n--- Effect Sizes (Cohen's d) ---")
    print(test_results['effect_sizes'].to_string())
    
    print("\n--- Dose-Response Regression ---")
    print(test_results['dose_response'].to_string())
    
    print("\n" + "=" * 80)
