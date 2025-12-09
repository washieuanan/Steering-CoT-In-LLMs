"""
Statistical Tests for Intervention Analysis

This module provides statistical testing functions for analyzing Phase B
intervention experiment results.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binom, mannwhitneyu, ttest_1samp, ttest_ind, wilcoxon

from . import metrics


# ==============================================================================
# McNemar's Test
# ==============================================================================

def mcnemar_test(wrong_to_right: int, right_to_wrong: int) -> Dict:
    """Run exact McNemar's test using binomial distribution.
    
    McNemar's test is appropriate for paired binary outcomes, testing whether
    the row and column marginal frequencies are equal (i.e., whether the
    intervention changes outcomes symmetrically).
    
    Args:
        wrong_to_right: Number of examples that went from wrong to right
        right_to_wrong: Number of examples that went from right to wrong
        
    Returns:
        Dictionary with test results
    """
    n = wrong_to_right + right_to_wrong
    
    if n == 0:
        return {
            'n_discordant': 0,
            'wrong_to_right': wrong_to_right,
            'right_to_wrong': right_to_wrong,
            'p_value': 1.0,
            'significant': False,
            'interpretation': 'No discordant pairs'
        }
    
    # Two-sided exact McNemar's test using binomial
    k = min(wrong_to_right, right_to_wrong)
    p_value = 2 * binom.cdf(k, n, 0.5)
    p_value = min(p_value, 1.0)
    
    # Interpretation
    if p_value < 0.05:
        if wrong_to_right > right_to_wrong:
            interpretation = 'Significant improvement (more fixes than breaks)'
        else:
            interpretation = 'Significant harm (more breaks than fixes)'
    else:
        interpretation = 'No significant difference'
    
    return {
        'n_discordant': n,
        'wrong_to_right': wrong_to_right,
        'right_to_wrong': right_to_wrong,
        'net_gain': wrong_to_right - right_to_wrong,
        'p_value': p_value,
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
        'interpretation': interpretation
    }


def run_mcnemar_test(df: pd.DataFrame, 
                     mode: Optional[str] = None,
                     alpha_filter: Optional[str] = None,
                     metric: str = 'answer') -> Dict:
    """Run McNemar's test on experiment results.
    
    Args:
        df: DataFrame with experiment results
        mode: Filter to specific mode (e.g., 'add', 'random')
        alpha_filter: 'positive' for α > 0, 'negative' for α < 0, None for all
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with McNemar test results
    """
    # Apply filters
    if mode and 'mode' in df.columns:
        df = df[df['mode'] == mode]
    
    if alpha_filter and 'alpha' in df.columns:
        if alpha_filter == 'positive':
            df = df[df['alpha'] > 0]
        elif alpha_filter == 'negative':
            df = df[df['alpha'] < 0]
    
    # Count flips
    w2r = metrics.count_wrong_to_right(df, metric)
    r2w = metrics.count_right_to_wrong(df, metric)
    
    result = mcnemar_test(w2r, r2w)
    result['mode'] = mode
    result['alpha_filter'] = alpha_filter
    result['n_total'] = len(df)
    
    return result


def run_mcnemar_by_group(df: pd.DataFrame,
                         group_cols: List[str] = ['mode', 'alpha'],
                         metric: str = 'answer') -> pd.DataFrame:
    """Run McNemar's test for each group.
    
    Args:
        df: DataFrame with experiment results
        group_cols: Columns to group by
        metric: 'answer' or 'reasoning'
        
    Returns:
        DataFrame with McNemar results per group
    """
    group_cols = [c for c in group_cols if c in df.columns]
    
    if not group_cols:
        result = run_mcnemar_test(df, metric=metric)
        return pd.DataFrame([result])
    
    results = []
    
    for name, group in df.groupby(group_cols):
        if not isinstance(name, tuple):
            name = (name,)
        
        w2r = metrics.count_wrong_to_right(group, metric)
        r2w = metrics.count_right_to_wrong(group, metric)
        
        result = mcnemar_test(w2r, r2w)
        result.update(dict(zip(group_cols, name)))
        result['n_total'] = len(group)
        
        results.append(result)
    
    return pd.DataFrame(results)


# ==============================================================================
# Sign Test
# ==============================================================================

def sign_test(positive: int, negative: int) -> Dict:
    """Run sign test.
    
    The sign test is a non-parametric test for matched pairs, testing whether
    positive and negative changes are equally likely.
    
    Args:
        positive: Number of positive changes
        negative: Number of negative changes
        
    Returns:
        Dictionary with test results
    """
    n_total = positive + negative
    
    if n_total == 0:
        return {
            'n_positive': positive,
            'n_negative': negative,
            'n_total': n_total,
            'p_value': 1.0,
            'significant': False
        }
    
    # Two-sided sign test
    k = min(positive, negative)
    p_value = 2 * binom.cdf(k, n_total, 0.5)
    p_value = min(p_value, 1.0)
    
    return {
        'n_positive': positive,
        'n_negative': negative,
        'n_total': n_total,
        'n_zero': 0,  # Placeholder, computed separately
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def run_sign_test(df: pd.DataFrame,
                  mode: Optional[str] = None,
                  alpha_filter: Optional[str] = None,
                  metric: str = 'answer') -> Dict:
    """Run sign test on delta values.
    
    Args:
        df: DataFrame with experiment results
        mode: Filter to specific mode
        alpha_filter: 'positive' for α > 0, 'negative' for α < 0, None for all
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with sign test results
    """
    # Apply filters
    if mode and 'mode' in df.columns:
        df = df[df['mode'] == mode]
    
    if alpha_filter and 'alpha' in df.columns:
        if alpha_filter == 'positive':
            df = df[df['alpha'] > 0]
        elif alpha_filter == 'negative':
            df = df[df['alpha'] < 0]
    
    # Compute per-example deltas
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return {'error': 'Required columns not found'}
    
    # Convert to numeric
    df = df.copy()
    for col in [base_col, intv_col]:
        if df[col].dtype == object:
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    deltas = df[intv_col].astype(int) - df[base_col].astype(int)
    
    n_positive = (deltas > 0).sum()
    n_negative = (deltas < 0).sum()
    n_zero = (deltas == 0).sum()
    
    result = sign_test(n_positive, n_negative)
    result['n_zero'] = n_zero
    result['mode'] = mode
    result['alpha_filter'] = alpha_filter
    
    return result


# ==============================================================================
# T-Tests
# ==============================================================================

def run_ttest_vs_zero(df: pd.DataFrame,
                      mode: Optional[str] = None,
                      alpha_filter: Optional[str] = None,
                      alternative: str = 'two-sided',
                      metric: str = 'answer') -> Dict:
    """Run one-sample t-test against zero.
    
    Tests whether the mean delta is significantly different from zero.
    
    Args:
        df: DataFrame with experiment results
        mode: Filter to specific mode
        alpha_filter: 'positive' for α > 0, 'negative' for α < 0, None for all
        alternative: 'two-sided', 'less', or 'greater'
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with t-test results
    """
    # Apply filters
    if mode and 'mode' in df.columns:
        df = df[df['mode'] == mode]
    
    if alpha_filter and 'alpha' in df.columns:
        if alpha_filter == 'positive':
            df = df[df['alpha'] > 0]
        elif alpha_filter == 'negative':
            df = df[df['alpha'] < 0]
    
    # Compute per-example deltas
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return {'error': 'Required columns not found'}
    
    # Convert to numeric
    df = df.copy()
    for col in [base_col, intv_col]:
        if df[col].dtype == object:
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    deltas = (df[intv_col].astype(int) - df[base_col].astype(int)).dropna()
    
    if len(deltas) < 2:
        return {
            'mean': np.nan,
            'std': np.nan,
            'n': len(deltas),
            't_stat': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    
    t_stat, p_value = ttest_1samp(deltas, 0, alternative=alternative)
    
    return {
        'mean': deltas.mean(),
        'std': deltas.std(),
        'n': len(deltas),
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mode': mode,
        'alpha_filter': alpha_filter,
        'alternative': alternative
    }


def run_ttest_two_groups(df: pd.DataFrame,
                         group1_mode: str = 'add',
                         group2_mode: str = 'random',
                         alpha_filter: Optional[str] = None,
                         metric: str = 'answer') -> Dict:
    """Run two-sample t-test comparing two modes.
    
    Args:
        df: DataFrame with experiment results
        group1_mode: First mode to compare
        group2_mode: Second mode to compare
        alpha_filter: 'positive' for α > 0, 'negative' for α < 0, None for all
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with t-test results
    """
    if 'mode' not in df.columns:
        return {'error': 'Mode column not found'}
    
    group1 = df[df['mode'] == group1_mode]
    group2 = df[df['mode'] == group2_mode]
    
    if alpha_filter and 'alpha' in df.columns:
        if alpha_filter == 'positive':
            group1 = group1[group1['alpha'] > 0]
            group2 = group2[group2['alpha'] > 0]
        elif alpha_filter == 'negative':
            group1 = group1[group1['alpha'] < 0]
            group2 = group2[group2['alpha'] < 0]
    
    # Compute deltas for each group
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return {'error': 'Required columns not found'}
    
    def compute_deltas(g):
        g = g.copy()
        for col in [base_col, intv_col]:
            if g[col].dtype == object:
                g[col] = g[col].map({'True': 1, 'False': 0, True: 1, False: 0})
        return (g[intv_col].astype(int) - g[base_col].astype(int)).dropna()
    
    deltas1 = compute_deltas(group1)
    deltas2 = compute_deltas(group2)
    
    if len(deltas1) < 2 or len(deltas2) < 2:
        return {
            'group1_mean': np.nan,
            'group2_mean': np.nan,
            't_stat': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    
    t_stat, p_value = ttest_ind(deltas1, deltas2)
    
    return {
        'group1': group1_mode,
        'group2': group2_mode,
        'group1_mean': deltas1.mean(),
        'group2_mean': deltas2.mean(),
        'group1_std': deltas1.std(),
        'group2_std': deltas2.std(),
        'group1_n': len(deltas1),
        'group2_n': len(deltas2),
        'difference': deltas1.mean() - deltas2.mean(),
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'alpha_filter': alpha_filter
    }


# ==============================================================================
# Non-Parametric Tests
# ==============================================================================

def run_wilcoxon_test(df: pd.DataFrame,
                      mode: Optional[str] = None,
                      alpha_filter: Optional[str] = None,
                      metric: str = 'answer') -> Dict:
    """Run Wilcoxon signed-rank test.
    
    Non-parametric alternative to one-sample t-test.
    
    Args:
        df: DataFrame with experiment results
        mode: Filter to specific mode
        alpha_filter: 'positive' for α > 0, 'negative' for α < 0, None for all
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with test results
    """
    # Apply filters
    if mode and 'mode' in df.columns:
        df = df[df['mode'] == mode]
    
    if alpha_filter and 'alpha' in df.columns:
        if alpha_filter == 'positive':
            df = df[df['alpha'] > 0]
        elif alpha_filter == 'negative':
            df = df[df['alpha'] < 0]
    
    # Compute per-example deltas
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return {'error': 'Required columns not found'}
    
    df = df.copy()
    for col in [base_col, intv_col]:
        if df[col].dtype == object:
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    deltas = (df[intv_col].astype(int) - df[base_col].astype(int)).dropna()
    deltas = deltas[deltas != 0]  # Remove zeros
    
    if len(deltas) < 2:
        return {
            'median': np.nan,
            'n': len(deltas),
            'W_stat': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    
    try:
        W_stat, p_value = wilcoxon(deltas, alternative='two-sided')
    except ValueError:
        return {
            'median': deltas.median(),
            'n': len(deltas),
            'W_stat': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    
    return {
        'median': deltas.median(),
        'mean': deltas.mean(),
        'n': len(deltas),
        'W_stat': W_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mode': mode,
        'alpha_filter': alpha_filter
    }


def run_mannwhitney_test(df: pd.DataFrame,
                         group1_mode: str = 'add',
                         group2_mode: str = 'random',
                         alpha_value: Optional[float] = None,
                         metric: str = 'answer') -> Dict:
    """Run Mann-Whitney U test comparing two modes.
    
    Non-parametric alternative to two-sample t-test.
    
    Args:
        df: DataFrame with experiment results
        group1_mode: First mode to compare
        group2_mode: Second mode to compare
        alpha_value: Filter to specific alpha value
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with test results
    """
    if 'mode' not in df.columns:
        return {'error': 'Mode column not found'}
    
    group1 = df[df['mode'] == group1_mode]
    group2 = df[df['mode'] == group2_mode]
    
    if alpha_value is not None and 'alpha' in df.columns:
        group1 = group1[group1['alpha'] == alpha_value]
        group2 = group2[group2['alpha'] == alpha_value]
    
    # Compute deltas
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return {'error': 'Required columns not found'}
    
    def compute_deltas(g):
        g = g.copy()
        for col in [base_col, intv_col]:
            if g[col].dtype == object:
                g[col] = g[col].map({'True': 1, 'False': 0, True: 1, False: 0})
        return (g[intv_col].astype(int) - g[base_col].astype(int)).dropna()
    
    deltas1 = compute_deltas(group1)
    deltas2 = compute_deltas(group2)
    
    if len(deltas1) < 2 or len(deltas2) < 2:
        return {
            'U_stat': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    
    U_stat, p_value = mannwhitneyu(deltas1, deltas2, alternative='two-sided')
    
    # Effect size (rank-biserial correlation)
    n1, n2 = len(deltas1), len(deltas2)
    r = 1 - (2 * U_stat) / (n1 * n2)
    
    return {
        'group1': group1_mode,
        'group2': group2_mode,
        'group1_mean': deltas1.mean(),
        'group2_mean': deltas2.mean(),
        'group1_median': deltas1.median(),
        'group2_median': deltas2.median(),
        'group1_n': n1,
        'group2_n': n2,
        'U_stat': U_stat,
        'p_value': p_value,
        'effect_size_r': r,
        'significant': p_value < 0.05,
        'alpha_value': alpha_value
    }


# ==============================================================================
# Effect Size Measures
# ==============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size.
    
    Cohen's d is a standardized measure of effect size.
    Interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large
    
    Args:
        group1: First group's values
        group2: Second group's values
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    
    if n1 < 2 or n2 < 2:
        return np.nan
    
    var1 = group1.var(ddof=1)
    var2 = group2.var(ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    return (group1.mean() - group2.mean()) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    if np.isnan(d):
        return 'N/A'
    
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'


def compute_effect_size(df: pd.DataFrame,
                        mode1: str = 'add',
                        mode2: str = 'random',
                        metric: str = 'answer') -> Dict:
    """Compute effect size (Cohen's d) between two modes.
    
    Args:
        df: DataFrame with experiment results
        mode1: First mode
        mode2: Second mode
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with effect size results
    """
    if 'mode' not in df.columns:
        return {'error': 'Mode column not found'}
    
    # Compute deltas for each mode
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return {'error': 'Required columns not found'}
    
    def compute_deltas(g):
        g = g.copy()
        for col in [base_col, intv_col]:
            if g[col].dtype == object:
                g[col] = g[col].map({'True': 1, 'False': 0, True: 1, False: 0})
        return (g[intv_col].astype(int) - g[base_col].astype(int)).dropna().values
    
    deltas1 = compute_deltas(df[df['mode'] == mode1])
    deltas2 = compute_deltas(df[df['mode'] == mode2])
    
    d = cohens_d(deltas1, deltas2)
    
    return {
        'mode1': mode1,
        'mode2': mode2,
        'mode1_mean': deltas1.mean() if len(deltas1) > 0 else np.nan,
        'mode2_mean': deltas2.mean() if len(deltas2) > 0 else np.nan,
        'cohens_d': d,
        'interpretation': interpret_cohens_d(d)
    }


def compute_all_effect_sizes(df: pd.DataFrame, metric: str = 'answer') -> pd.DataFrame:
    """Compute effect sizes for all mode/dataset combinations.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        DataFrame with effect sizes
    """
    results = []
    
    # Overall
    overall = compute_effect_size(df, 'add', 'random', metric)
    overall['group'] = 'overall'
    results.append(overall)
    
    # By dataset if available
    if 'dataset' in df.columns:
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            es = compute_effect_size(dataset_df, 'add', 'random', metric)
            es['group'] = f'dataset_{dataset}'
            results.append(es)
    
    # By alpha if available
    if 'alpha' in df.columns:
        for alpha in df['alpha'].unique():
            alpha_df = df[df['alpha'] == alpha]
            es = compute_effect_size(alpha_df, 'add', 'random', metric)
            es['group'] = f'alpha_{alpha}'
            results.append(es)
    
    return pd.DataFrame(results)


# ==============================================================================
# Dose-Response Analysis
# ==============================================================================

def dose_response_regression(df: pd.DataFrame,
                            mode: str = 'add',
                            metric: str = 'answer') -> Dict:
    """Fit linear regression of delta ~ alpha.
    
    Tests for a dose-response relationship between intervention strength
    and effect size.
    
    Args:
        df: DataFrame with experiment results
        mode: Intervention mode to analyze
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with regression results
    """
    if 'mode' in df.columns:
        df = df[df['mode'] == mode]
    
    if 'alpha' not in df.columns:
        return {'error': 'Alpha column not found'}
    
    # Compute delta for each example
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return {'error': 'Required columns not found'}
    
    df = df.copy()
    for col in [base_col, intv_col]:
        if df[col].dtype == object:
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    alphas = df['alpha'].values
    deltas = (df[intv_col].astype(int) - df[base_col].astype(int)).values
    
    # Remove NaN
    mask = ~np.isnan(deltas)
    alphas = alphas[mask]
    deltas = deltas[mask]
    
    if len(alphas) < 3:
        return {
            'slope': np.nan,
            'intercept': np.nan,
            'r_squared': np.nan,
            'p_value': 1.0
        }
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(alphas, deltas)
    
    return {
        'mode': mode,
        'slope': slope,
        'slope_se': std_err,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n': len(alphas)
    }


# ==============================================================================
# Bootstrap Confidence Intervals
# ==============================================================================

def bootstrap_ci(data: np.ndarray,
                 statistic: callable = np.mean,
                 n_bootstrap: int = 1000,
                 ci: float = 0.95,
                 seed: Optional[int] = None) -> Tuple[float, float]:
    """Compute bootstrap confidence interval.
    
    Args:
        data: Input data array
        statistic: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default: 0.95)
        seed: Random seed
        
    Returns:
        Tuple of (lower, upper) confidence bounds
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    if n == 0:
        return (np.nan, np.nan)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    
    # Percentile method
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    
    return (lower, upper)


def compute_bootstrap_ci(df: pd.DataFrame,
                         mode: Optional[str] = None,
                         metric: str = 'answer',
                         n_bootstrap: int = 1000,
                         ci: float = 0.95) -> Dict:
    """Compute bootstrap confidence interval for delta accuracy.
    
    Args:
        df: DataFrame with experiment results
        mode: Filter to specific mode
        metric: 'answer' or 'reasoning'
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level
        
    Returns:
        Dictionary with CI results
    """
    if mode and 'mode' in df.columns:
        df = df[df['mode'] == mode]
    
    # Compute deltas
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return {'error': 'Required columns not found'}
    
    df = df.copy()
    for col in [base_col, intv_col]:
        if df[col].dtype == object:
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    deltas = (df[intv_col].astype(int) - df[base_col].astype(int)).dropna().values
    
    if len(deltas) == 0:
        return {'error': 'No data'}
    
    mean = deltas.mean()
    lower, upper = bootstrap_ci(deltas, np.mean, n_bootstrap, ci)
    
    return {
        'mean': mean,
        'ci_lower': lower,
        'ci_upper': upper,
        'ci_level': ci,
        'n': len(deltas),
        'mode': mode
    }


# ==============================================================================
# Comprehensive Testing
# ==============================================================================

def run_all_tests(df: pd.DataFrame, metric: str = 'answer') -> Dict[str, pd.DataFrame]:
    """Run all statistical tests and return results.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary mapping test names to result DataFrames
    """
    results = {}
    
    # McNemar tests
    mcnemar_results = []
    for mode in ['add', 'random']:
        for alpha_filter in ['positive', 'negative', None]:
            result = run_mcnemar_test(df, mode=mode, alpha_filter=alpha_filter, metric=metric)
            result['condition'] = f'{mode}_{alpha_filter or "all"}'
            mcnemar_results.append(result)
    results['mcnemar'] = pd.DataFrame(mcnemar_results)
    
    # Sign tests
    sign_results = []
    for mode in ['add', 'random']:
        for alpha_filter in ['positive', 'negative', None]:
            result = run_sign_test(df, mode=mode, alpha_filter=alpha_filter, metric=metric)
            result['condition'] = f'{mode}_{alpha_filter or "all"}'
            sign_results.append(result)
    results['sign_test'] = pd.DataFrame(sign_results)
    
    # T-tests
    ttest_results = []
    for mode in ['add', 'random']:
        for alpha_filter in ['positive', 'negative', None]:
            result = run_ttest_vs_zero(df, mode=mode, alpha_filter=alpha_filter, metric=metric)
            result['condition'] = f'{mode}_{alpha_filter or "all"}'
            ttest_results.append(result)
    results['ttest'] = pd.DataFrame(ttest_results)
    
    # Mann-Whitney
    mw_results = []
    if 'alpha' in df.columns:
        for alpha in df['alpha'].unique():
            result = run_mannwhitney_test(df, alpha_value=alpha, metric=metric)
            mw_results.append(result)
    results['mann_whitney'] = pd.DataFrame(mw_results)
    
    # Effect sizes
    results['effect_sizes'] = compute_all_effect_sizes(df, metric)
    
    # Dose-response
    dr_results = []
    for mode in ['add', 'random']:
        result = dose_response_regression(df, mode=mode, metric=metric)
        dr_results.append(result)
    results['dose_response'] = pd.DataFrame(dr_results)
    
    return results


def print_test_summary(results: Dict[str, pd.DataFrame]):
    """Print formatted summary of statistical test results.
    
    Args:
        results: Dictionary of test results from run_all_tests()
    """
    print("\n" + "=" * 80)
    print("STATISTICAL TEST RESULTS SUMMARY")
    print("=" * 80)
    
    for test_name, df in results.items():
        print(f"\n--- {test_name.upper()} ---")
        if len(df) > 0:
            print(df.to_string())
        else:
            print("No results")
    
    print("\n" + "=" * 80)
