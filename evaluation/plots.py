"""
Visualization Functions for Intervention Analysis

This module provides plotting functions for visualizing Phase B intervention
experiment results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import metrics


# ==============================================================================
# Style Configuration
# ==============================================================================

# Color palettes
MODE_COLORS = {
    'add': '#2ecc71',      # Green
    'random': '#e74c3c',   # Red
    'lesion': '#9b59b6',   # Purple
    'rescue': '#3498db',   # Blue
}

LOCALITY_COLORS = {
    'cot': '#3498db',      # Blue
    'answer': '#e67e22',   # Orange
    'full': '#1abc9c',     # Teal
}

DATASET_COLORS = {
    'arc': '#3498db',
    'gsm8k': '#e74c3c',
    'mmlu_pro': '#9b59b6',
}


def setup_style():
    """Set up matplotlib style for publication-quality figures."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
    })


# ==============================================================================
# Alpha Sweep Plots
# ==============================================================================

def plot_alpha_sweep(df: pd.DataFrame,
                     metric: str = 'answer',
                     title: Optional[str] = None,
                     save_path: Optional[Union[str, Path]] = None,
                     figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot delta accuracy vs alpha for different modes.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and Axes objects
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'mode' not in df.columns or 'alpha' not in df.columns:
        ax.text(0.5, 0.5, 'Insufficient data for alpha sweep',
                transform=ax.transAxes, ha='center', va='center')
        return fig, ax
    
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        if len(mode_df) == 0:
            continue
        
        # Aggregate by alpha
        agg = metrics.aggregate_by_config(mode_df, ['alpha'], metric)
        if len(agg) == 0:
            continue
        
        color = MODE_COLORS.get(mode, 'gray')
        
        ax.plot(agg['alpha'], agg['delta'], 'o-', 
                label=mode.capitalize(), color=color, linewidth=2, markersize=8)
        
        # Add error bands if we have multiple examples per alpha
        if agg['n'].min() > 1:
            # Compute standard error
            se = []
            for alpha in agg['alpha']:
                alpha_df = mode_df[mode_df['alpha'] == alpha]
                deltas = []
                for _, row in alpha_df.iterrows():
                    base_correct = row.get(f'baseline_{metric}_correct', False)
                    intv_correct = row.get(f'intv_{metric}_correct', False)
                    # Convert string to bool if needed
                    if isinstance(base_correct, str):
                        base_correct = base_correct == 'True'
                    if isinstance(intv_correct, str):
                        intv_correct = intv_correct == 'True'
                    deltas.append(int(intv_correct) - int(base_correct))
                if len(deltas) > 1:
                    se.append(np.std(deltas) / np.sqrt(len(deltas)))
                else:
                    se.append(0)
            
            se = np.array(se)
            ax.fill_between(agg['alpha'], agg['delta'] - 1.96 * se, 
                           agg['delta'] + 1.96 * se, alpha=0.2, color=color)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Alpha (Intervention Strength)')
    ax.set_ylabel(f'Δ {metric.capitalize()} Accuracy')
    ax.set_title(title or f'Effect of Intervention Strength on {metric.capitalize()} Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_alpha_sweep_by_layer(df: pd.DataFrame,
                              mode: str = 'add',
                              metric: str = 'answer',
                              save_path: Optional[Union[str, Path]] = None,
                              figsize: Tuple[int, int] = (12, 5)) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot alpha sweep faceted by layer.
    
    Args:
        df: DataFrame with experiment results
        mode: Intervention mode to plot
        metric: 'answer' or 'reasoning'
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and list of Axes objects
    """
    setup_style()
    
    if 'mode' in df.columns:
        df = df[df['mode'] == mode]
    
    if 'layer' not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No layer data available',
                transform=ax.transAxes, ha='center', va='center')
        return fig, [ax]
    
    layers = sorted(df['layer'].unique())
    n_layers = len(layers)
    
    fig, axes = plt.subplots(1, n_layers, figsize=figsize, sharey=True)
    if n_layers == 1:
        axes = [axes]
    
    for ax, layer in zip(axes, layers):
        layer_df = df[df['layer'] == layer]
        
        if 'alpha' not in layer_df.columns:
            continue
        
        agg = metrics.aggregate_by_config(layer_df, ['alpha'], metric)
        if len(agg) == 0:
            continue
        
        color = MODE_COLORS.get(mode, '#2ecc71')
        ax.plot(agg['alpha'], agg['delta'], 'o-', color=color, linewidth=2, markersize=8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Alpha')
        ax.set_title(f'Layer {layer}')
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel(f'Δ {metric.capitalize()} Accuracy')
    fig.suptitle(f'Alpha Sweep by Layer ({mode.capitalize()} Mode)', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


# ==============================================================================
# Heatmaps
# ==============================================================================

def plot_delta_heatmap(df: pd.DataFrame,
                       rows: str = 'layer',
                       cols: str = 'alpha',
                       mode: str = 'add',
                       metric: str = 'answer',
                       title: Optional[str] = None,
                       save_path: Optional[Union[str, Path]] = None,
                       figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot heatmap of delta accuracy.
    
    Args:
        df: DataFrame with experiment results
        rows: Column to use for heatmap rows
        cols: Column to use for heatmap columns
        mode: Intervention mode to plot
        metric: 'answer' or 'reasoning'
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and Axes objects
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'mode' in df.columns:
        df = df[df['mode'] == mode]
    
    if rows not in df.columns or cols not in df.columns:
        ax.text(0.5, 0.5, 'Insufficient data for heatmap',
                transform=ax.transAxes, ha='center', va='center')
        return fig, ax
    
    # Aggregate
    agg = metrics.aggregate_by_config(df, [rows, cols], metric)
    
    if len(agg) == 0:
        ax.text(0.5, 0.5, 'No data available',
                transform=ax.transAxes, ha='center', va='center')
        return fig, ax
    
    # Pivot for heatmap
    pivot = agg.pivot(index=rows, columns=cols, values='delta')
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=ax, cbar_kws={'label': f'Δ {metric.capitalize()} Accuracy'})
    
    ax.set_title(title or f'Delta {metric.capitalize()} Accuracy ({mode.capitalize()} Mode)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_specificity_heatmap(df: pd.DataFrame,
                             rows: str = 'layer',
                             cols: str = 'alpha',
                             metric: str = 'answer',
                             save_path: Optional[Union[str, Path]] = None,
                             figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot heatmap of specificity (add - random) effect.
    
    Args:
        df: DataFrame with experiment results
        rows: Column to use for heatmap rows
        cols: Column to use for heatmap columns
        metric: 'answer' or 'reasoning'
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and Axes objects
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compare modes
    comparison = metrics.compare_modes(df, 'add', 'random', metric)
    
    if len(comparison) == 0:
        ax.text(0.5, 0.5, 'Insufficient data for specificity heatmap',
                transform=ax.transAxes, ha='center', va='center')
        return fig, ax
    
    if rows not in comparison.columns or cols not in comparison.columns:
        ax.text(0.5, 0.5, 'Insufficient data for specificity heatmap',
                transform=ax.transAxes, ha='center', va='center')
        return fig, ax
    
    # Pivot for heatmap
    pivot = comparison.pivot(index=rows, columns=cols, values='difference')
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=ax, cbar_kws={'label': 'Specificity (Add - Random)'})
    
    ax.set_title('Specificity: Add Mode vs Random Mode\n(Positive = Add is better)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# ==============================================================================
# Flip Analysis Plots
# ==============================================================================

def plot_flip_bars(df: pd.DataFrame,
                   group_by: str = 'alpha',
                   metric: str = 'answer',
                   save_path: Optional[Union[str, Path]] = None,
                   figsize: Tuple[int, int] = (12, 5)) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot stacked bar chart of flips by mode.
    
    Args:
        df: DataFrame with experiment results
        group_by: Column to group bars by ('alpha' or 'layer')
        metric: 'answer' or 'reasoning'
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and list of Axes objects
    """
    setup_style()
    
    modes = df['mode'].unique() if 'mode' in df.columns else ['all']
    n_modes = len(modes)
    
    fig, axes = plt.subplots(1, n_modes, figsize=figsize, sharey=True)
    if n_modes == 1:
        axes = [axes]
    
    for ax, mode in zip(axes, modes):
        mode_df = df if mode == 'all' else df[df['mode'] == mode]
        
        if group_by not in mode_df.columns:
            continue
        
        # Aggregate by group
        agg = metrics.aggregate_by_config(mode_df, [group_by], metric)
        
        if len(agg) == 0:
            continue
        
        x = np.arange(len(agg))
        width = 0.35
        
        ax.bar(x - width/2, agg['wrong_to_right'], width, 
               label='Wrong→Right', color='#2ecc71')
        ax.bar(x + width/2, agg['right_to_wrong'], width,
               label='Right→Wrong', color='#e74c3c')
        
        ax.set_xlabel(group_by.capitalize())
        ax.set_ylabel('Count')
        ax.set_title(f'{mode.capitalize()} Mode')
        ax.set_xticks(x)
        ax.set_xticklabels(agg[group_by].astype(str), rotation=45)
        ax.legend()
    
    fig.suptitle('Flip Analysis', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_net_gain(df: pd.DataFrame,
                  metric: str = 'answer',
                  save_path: Optional[Union[str, Path]] = None,
                  figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot net gain (wrong→right - right→wrong) by alpha.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and Axes objects
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'mode' not in df.columns or 'alpha' not in df.columns:
        ax.text(0.5, 0.5, 'Insufficient data',
                transform=ax.transAxes, ha='center', va='center')
        return fig, ax
    
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        
        agg = metrics.aggregate_by_config(mode_df, ['alpha'], metric)
        if len(agg) == 0:
            continue
        
        color = MODE_COLORS.get(mode, 'gray')
        ax.plot(agg['alpha'], agg['net_gain'], 'o-',
                label=mode.capitalize(), color=color, linewidth=2, markersize=8)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Net Gain (Wrong→Right - Right→Wrong)')
    ax.set_title('Net Benefit of Intervention')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# ==============================================================================
# Distribution Plots
# ==============================================================================

def plot_delta_distribution(df: pd.DataFrame,
                            metric: str = 'answer',
                            save_path: Optional[Union[str, Path]] = None,
                            figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot distribution of delta values by mode.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and Axes objects
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute per-example deltas
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        ax.text(0.5, 0.5, 'Insufficient data',
                transform=ax.transAxes, ha='center', va='center')
        return fig, ax
    
    # Convert to numeric
    df = df.copy()
    for col in [base_col, intv_col]:
        if df[col].dtype == object:
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    df['delta'] = df[intv_col].astype(int) - df[base_col].astype(int)
    
    if 'mode' in df.columns:
        for mode in df['mode'].unique():
            mode_df = df[df['mode'] == mode]
            color = MODE_COLORS.get(mode, 'gray')
            ax.hist(mode_df['delta'], bins=[-1.5, -0.5, 0.5, 1.5], 
                   alpha=0.5, label=mode.capitalize(), color=color, edgecolor='black')
    else:
        ax.hist(df['delta'], bins=[-1.5, -0.5, 0.5, 1.5], 
               alpha=0.7, color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Delta (Intervention - Baseline)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {metric.capitalize()} Changes')
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['Right→Wrong', 'No Change', 'Wrong→Right'])
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_accuracy_comparison(df: pd.DataFrame,
                             metric: str = 'answer',
                             save_path: Optional[Union[str, Path]] = None,
                             figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot baseline vs intervention accuracy comparison.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and Axes objects
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'mode' not in df.columns:
        ax.text(0.5, 0.5, 'No mode data available',
                transform=ax.transAxes, ha='center', va='center')
        return fig, ax
    
    modes = df['mode'].unique()
    x = np.arange(len(modes))
    width = 0.35
    
    base_accs = []
    intv_accs = []
    
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        base_accs.append(metrics.compute_baseline_accuracy(mode_df, metric))
        intv_accs.append(metrics.compute_intervention_accuracy(mode_df, metric))
    
    ax.bar(x - width/2, base_accs, width, label='Baseline', color='#3498db')
    ax.bar(x + width/2, intv_accs, width, label='Intervention', color='#e67e22')
    
    ax.set_xlabel('Mode')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{metric.capitalize()} Accuracy: Baseline vs Intervention')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modes])
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# ==============================================================================
# Locality Comparison Plots
# ==============================================================================

def plot_locality_comparison(df: pd.DataFrame,
                             localities: List[str] = ['cot', 'answer'],
                             metric: str = 'answer',
                             save_path: Optional[Union[str, Path]] = None,
                             figsize: Tuple[int, int] = (12, 5)) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot comparison between different localities.
    
    Args:
        df: DataFrame with experiment results from multiple localities
        localities: List of localities to compare
        metric: 'answer' or 'reasoning'
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and list of Axes objects
    """
    setup_style()
    
    if 'locality' not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No locality data available',
                transform=ax.transAxes, ha='center', va='center')
        return fig, [ax]
    
    n_loc = len(localities)
    fig, axes = plt.subplots(1, n_loc, figsize=figsize, sharey=True)
    if n_loc == 1:
        axes = [axes]
    
    for ax, locality in zip(axes, localities):
        loc_df = df[df['locality'] == locality]
        
        if len(loc_df) == 0:
            ax.text(0.5, 0.5, f'No data for {locality}',
                    transform=ax.transAxes, ha='center', va='center')
            continue
        
        # Plot alpha sweep for this locality
        if 'mode' in loc_df.columns and 'alpha' in loc_df.columns:
            for mode in loc_df['mode'].unique():
                mode_df = loc_df[loc_df['mode'] == mode]
                agg = metrics.aggregate_by_config(mode_df, ['alpha'], metric)
                if len(agg) > 0:
                    color = MODE_COLORS.get(mode, 'gray')
                    ax.plot(agg['alpha'], agg['delta'], 'o-',
                           label=mode.capitalize(), color=color, linewidth=2)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Alpha')
        ax.set_title(f'{locality.upper()} Locality')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel(f'Δ {metric.capitalize()} Accuracy')
    fig.suptitle('Locality Comparison', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_locality_difference(df: pd.DataFrame,
                             locality1: str = 'cot',
                             locality2: str = 'answer',
                             metric: str = 'answer',
                             save_path: Optional[Union[str, Path]] = None,
                             figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the difference in effect between two localities.
    
    Args:
        df: DataFrame with experiment results from multiple localities
        locality1: First locality
        locality2: Second locality
        metric: 'answer' or 'reasoning'
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and Axes objects
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compare localities
    comparison = metrics.compare_localities(df, locality1, locality2, metric)
    
    if len(comparison) == 0:
        ax.text(0.5, 0.5, 'Insufficient data for comparison',
                transform=ax.transAxes, ha='center', va='center')
        return fig, ax
    
    # Plot difference by alpha if available
    if 'alpha' in comparison.columns:
        ax.bar(comparison['alpha'].astype(str), comparison['difference'],
              color='steelblue', edgecolor='black')
        ax.set_xlabel('Alpha')
    else:
        ax.bar(['Overall'], [comparison['difference'].iloc[0]],
              color='steelblue', edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel(f'Δ Effect ({locality1} - {locality2})')
    ax.set_title(f'Locality Comparison: {locality1.upper()} vs {locality2.upper()}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# ==============================================================================
# Multi-panel Summary Plots
# ==============================================================================

def plot_summary_dashboard(df: pd.DataFrame,
                           metric: str = 'answer',
                           save_path: Optional[Union[str, Path]] = None,
                           figsize: Tuple[int, int] = (15, 10)) -> Tuple[plt.Figure, np.ndarray]:
    """Create a multi-panel summary dashboard.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure and array of Axes objects
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Panel 1: Alpha sweep
    ax = axes[0, 0]
    if 'mode' in df.columns and 'alpha' in df.columns:
        for mode in df['mode'].unique():
            mode_df = df[df['mode'] == mode]
            agg = metrics.aggregate_by_config(mode_df, ['alpha'], metric)
            if len(agg) > 0:
                color = MODE_COLORS.get(mode, 'gray')
                ax.plot(agg['alpha'], agg['delta'], 'o-',
                       label=mode.capitalize(), color=color, linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Alpha')
        ax.set_ylabel(f'Δ Accuracy')
        ax.set_title('Alpha Sweep')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Panel 2: Accuracy comparison
    ax = axes[0, 1]
    if 'mode' in df.columns:
        modes = df['mode'].unique()
        x = np.arange(len(modes))
        width = 0.35
        base_accs = [metrics.compute_baseline_accuracy(df[df['mode'] == m], metric) for m in modes]
        intv_accs = [metrics.compute_intervention_accuracy(df[df['mode'] == m], metric) for m in modes]
        ax.bar(x - width/2, base_accs, width, label='Baseline', color='#3498db')
        ax.bar(x + width/2, intv_accs, width, label='Intervention', color='#e67e22')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modes])
        ax.set_ylabel('Accuracy')
        ax.set_title('Baseline vs Intervention')
        ax.legend()
        ax.set_ylim(0, 1)
    
    # Panel 3: Flip analysis
    ax = axes[1, 0]
    if 'mode' in df.columns:
        modes = df['mode'].unique()
        x = np.arange(len(modes))
        width = 0.35
        w2r = [metrics.count_wrong_to_right(df[df['mode'] == m], metric) for m in modes]
        r2w = [metrics.count_right_to_wrong(df[df['mode'] == m], metric) for m in modes]
        ax.bar(x - width/2, w2r, width, label='Wrong→Right', color='#2ecc71')
        ax.bar(x + width/2, r2w, width, label='Right→Wrong', color='#e74c3c')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modes])
        ax.set_ylabel('Count')
        ax.set_title('Flip Analysis')
        ax.legend()
    
    # Panel 4: Delta distribution
    ax = axes[1, 1]
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    if base_col in df.columns and intv_col in df.columns:
        df_temp = df.copy()
        for col in [base_col, intv_col]:
            if df_temp[col].dtype == object:
                df_temp[col] = df_temp[col].map({'True': 1, 'False': 0, True: 1, False: 0})
        df_temp['delta'] = df_temp[intv_col].astype(int) - df_temp[base_col].astype(int)
        
        delta_counts = df_temp['delta'].value_counts().sort_index()
        ax.bar(delta_counts.index, delta_counts.values, color='steelblue', edgecolor='black')
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['R→W', 'No Change', 'W→R'])
        ax.set_xlabel('Delta')
        ax.set_ylabel('Count')
        ax.set_title('Delta Distribution')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


# ==============================================================================
# Utility Functions
# ==============================================================================

def save_all_plots(df: pd.DataFrame,
                   output_dir: Union[str, Path],
                   metric: str = 'answer',
                   format: str = 'pdf'):
    """Generate and save all standard plots.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
        metric: 'answer' or 'reasoning'
        format: Output format ('pdf', 'png', 'svg')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Alpha sweep
    plot_alpha_sweep(df, metric, save_path=output_dir / f'alpha_sweep.{format}')
    plt.close()
    
    # Delta heatmap
    for mode in df['mode'].unique() if 'mode' in df.columns else ['all']:
        plot_delta_heatmap(df, mode=mode, metric=metric,
                          save_path=output_dir / f'heatmap_{mode}.{format}')
        plt.close()
    
    # Specificity heatmap
    plot_specificity_heatmap(df, metric=metric,
                            save_path=output_dir / f'specificity_heatmap.{format}')
    plt.close()
    
    # Flip bars
    plot_flip_bars(df, metric=metric, save_path=output_dir / f'flip_bars.{format}')
    plt.close()
    
    # Net gain
    plot_net_gain(df, metric=metric, save_path=output_dir / f'net_gain.{format}')
    plt.close()
    
    # Summary dashboard
    plot_summary_dashboard(df, metric=metric,
                          save_path=output_dir / f'summary_dashboard.{format}')
    plt.close()
    
    print(f"Saved all plots to {output_dir}")
