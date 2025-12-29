# tasks/bio_experiments/analyze_16bit.py
"""
Comprehensive analysis for 16-bit parity Bio-CTM ablation study.
Generates publication-ready figures and statistical analysis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_all_results(output_dir: str) -> Dict:
    """Load all experiment results from output directory."""
    results = {}
    
    for exp_name in sorted(os.listdir(output_dir)):
        exp_path = os.path.join(output_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
            
        results_file = os.path.join(exp_path, 'results.json')
        config_file = os.path.join(exp_path, 'config.json')
        
        if os.path.exists(results_file) and os.path.exists(config_file):
            with open(results_file) as f:
                exp_results = json.load(f)
            with open(config_file) as f:
                config = json.load(f)
            
            results[exp_name] = {
                'results': exp_results,
                'config': config
            }
    
    return results


def extract_condition(exp_name: str) -> Tuple[str, int]:
    """Extract condition name and seed from experiment name."""
    parts = exp_name.rsplit('_seed', 1)
    if len(parts) == 2:
        return parts[0], int(parts[1])
    return exp_name, 0


def aggregate_by_condition(results: Dict) -> Dict:
    """Aggregate results by experimental condition."""
    aggregated = defaultdict(lambda: {
        'test_acc': [], 'test_acc_curves': [],
        'train_acc': [], 'train_acc_curves': [],
        'test_loss': [], 'train_loss': [],
        'best_test_acc': [], 'final_test_acc': [],
        'seeds': [], 'configs': []
    })
    
    for exp_name, data in results.items():
        condition, seed = extract_condition(exp_name)
        res = data['results']
        
        aggregated[condition]['seeds'].append(seed)
        aggregated[condition]['configs'].append(data['config'])
        
        if res.get('test_acc'):
            aggregated[condition]['test_acc_curves'].append(res['test_acc'])
            aggregated[condition]['test_acc'].append(max(res['test_acc']))
            aggregated[condition]['final_test_acc'].append(res['test_acc'][-1])
        
        if res.get('train_acc'):
            aggregated[condition]['train_acc_curves'].append(res['train_acc'])
            aggregated[condition]['train_acc'].append(max(res['train_acc']))
        
        if res.get('test_loss'):
            aggregated[condition]['test_loss'].append(res['test_loss'][-1])
        
        if res.get('train_loss'):
            aggregated[condition]['train_loss'].append(res['train_loss'][-1])
        
        if res.get('best_test_acc'):
            aggregated[condition]['best_test_acc'].append(res['best_test_acc'])
    
    return dict(aggregated)


def compute_statistics(aggregated: Dict) -> pd.DataFrame:
    """Compute summary statistics for each condition."""
    rows = []
    
    baseline_accs = aggregated.get('baseline', {}).get('final_test_acc', [])
    
    for condition, data in aggregated.items():
        test_accs = data.get('final_test_acc', data.get('test_acc', []))
        
        if not test_accs:
            continue
        
        row = {
            'condition': condition,
            'n_seeds': len(test_accs),
            'test_acc_mean': np.mean(test_accs),
            'test_acc_std': np.std(test_accs),
            'test_acc_min': np.min(test_accs),
            'test_acc_max': np.max(test_accs),
            'train_acc_mean': np.mean(data['train_acc']) if data['train_acc'] else 0,
            'train_acc_std': np.std(data['train_acc']) if data['train_acc'] else 0,
        }
        
        # Statistical comparison vs baseline
        if condition != 'baseline' and len(baseline_accs) >= 2 and len(test_accs) >= 2:
            t_stat, p_val = stats.ttest_ind(test_accs, baseline_accs)
            row['vs_baseline_t'] = t_stat
            row['vs_baseline_p'] = p_val
            row['vs_baseline_significant'] = p_val < 0.05
            row['improvement_over_baseline'] = row['test_acc_mean'] - np.mean(baseline_accs)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(test_accs)**2 + np.std(baseline_accs)**2) / 2)
            if pooled_std > 0:
                row['cohens_d'] = (row['test_acc_mean'] - np.mean(baseline_accs)) / pooled_std
            else:
                row['cohens_d'] = 0
        else:
            row['vs_baseline_t'] = np.nan
            row['vs_baseline_p'] = np.nan
            row['vs_baseline_significant'] = False
            row['improvement_over_baseline'] = 0
            row['cohens_d'] = 0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('test_acc_mean', ascending=False)
    return df


def plot_learning_curves(aggregated: Dict, output_dir: str):
    """Plot learning curves with confidence bands."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Define condition groups for coloring
    condition_colors = {
        'baseline': '#7f7f7f',
        'full_bio': '#2ca02c',
        'refract_only': '#d62728',
        'refract_lateral': '#ff7f0e',
        'refract_lateral_homeo': '#9467bd',
    }
    
    # Test accuracy
    ax = axes[0]
    for condition, data in sorted(aggregated.items()):
        curves = data['test_acc_curves']
        if not curves:
            continue
        
        # Align curves to same length
        min_len = min(len(c) for c in curves)
        curves_aligned = np.array([c[:min_len] for c in curves])
        
        mean_curve = np.mean(curves_aligned, axis=0)
        std_curve = np.std(curves_aligned, axis=0)
        epochs = np.arange(1, min_len + 1)
        
        color = condition_colors.get(condition, None)
        alpha = 0.8 if condition in condition_colors else 0.4
        linewidth = 2.5 if condition in condition_colors else 1.5
        
        line = ax.plot(epochs, mean_curve, label=condition, linewidth=linewidth, alpha=alpha, color=color)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, 
                       alpha=0.15, color=line[0].get_color())
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy Learning Curves', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)
    
    # Train accuracy
    ax = axes[1]
    for condition, data in sorted(aggregated.items()):
        curves = data['train_acc_curves']
        if not curves:
            continue
        
        min_len = min(len(c) for c in curves)
        curves_aligned = np.array([c[:min_len] for c in curves])
        
        mean_curve = np.mean(curves_aligned, axis=0)
        epochs = np.arange(1, min_len + 1)
        
        color = condition_colors.get(condition, None)
        alpha = 0.8 if condition in condition_colors else 0.4
        
        ax.plot(epochs, mean_curve, label=condition, linewidth=1.5, alpha=alpha, color=color)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Accuracy', fontsize=12)
    ax.set_title('Training Accuracy Learning Curves', fontsize=14)
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: learning_curves.png")


def plot_bar_comparison(stats_df: pd.DataFrame, aggregated: Dict, output_dir: str):
    """Create bar plot comparing all conditions."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sort by mean accuracy
    df = stats_df.sort_values('test_acc_mean', ascending=True)
    
    conditions = df['condition'].tolist()
    means = df['test_acc_mean'].tolist()
    stds = df['test_acc_std'].tolist()
    
    y_pos = np.arange(len(conditions))
    
    # Color coding
    colors = []
    for cond in conditions:
        if cond == 'baseline':
            colors.append('#7f7f7f')
        elif cond == 'full_bio':
            colors.append('#2ca02c')
        elif 'refract' in cond and 'only' in cond:
            colors.append('#d62728')
        elif 'refract' in cond:
            colors.append('#ff7f0e')
        elif '_only' in cond:
            colors.append('#1f77b4')
        elif 'full_minus' in cond:
            colors.append('#9467bd')
        else:
            colors.append('#17becf')
    
    bars = ax.barh(y_pos, means, xerr=stds, capsize=3, color=colors, alpha=0.8, height=0.7)
    
    # Add individual seed points
    for i, cond in enumerate(conditions):
        if cond in aggregated:
            seed_accs = aggregated[cond].get('final_test_acc', aggregated[cond].get('test_acc', []))
            ax.scatter(seed_accs, [i] * len(seed_accs), color='black', s=30, zorder=5, alpha=0.6)
    
    # Add significance markers
    baseline_mean = stats_df[stats_df['condition'] == 'baseline']['test_acc_mean'].values
    if len(baseline_mean) > 0:
        ax.axvline(x=baseline_mean[0], color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
        
        for i, (_, row) in enumerate(df.iterrows()):
            if row.get('vs_baseline_significant', False):
                marker = '***' if row['vs_baseline_p'] < 0.001 else '**' if row['vs_baseline_p'] < 0.01 else '*'
                ax.text(row['test_acc_mean'] + row['test_acc_std'] + 0.01, i, marker, 
                       va='center', fontsize=14, color='red', fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(conditions)
    ax.set_xlabel('Test Accuracy', fontsize=12)
    ax.set_title('16-bit Parity: Bio-CTM Ablation Results', fontsize=14)
    ax.set_xlim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend for significance
    ax.text(0.98, 0.02, '* p<0.05  ** p<0.01  *** p<0.001\n(vs baseline)', 
           transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: bar_comparison.png")


def plot_mechanism_contribution(stats_df: pd.DataFrame, output_dir: str):
    """Analyze individual mechanism contributions."""
    
    # Extract single mechanism results
    single_mechs = stats_df[stats_df['condition'].str.endswith('_only')].copy()
    baseline = stats_df[stats_df['condition'] == 'baseline']
    
    if single_mechs.empty or baseline.empty:
        print("  Skipping mechanism contribution plot (insufficient data)")
        return
    
    baseline_acc = baseline['test_acc_mean'].values[0]
    
    # Calculate improvement over baseline
    single_mechs = single_mechs.copy()
    single_mechs['improvement'] = single_mechs['test_acc_mean'] - baseline_acc
    single_mechs['mech_name'] = single_mechs['condition'].str.replace('_only', '')
    
    # Sort by improvement
    single_mechs = single_mechs.sort_values('improvement', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#d62728' if imp > 0 else '#1f77b4' for imp in single_mechs['improvement']]
    
    y_pos = np.arange(len(single_mechs))
    bars = ax.barh(y_pos, single_mechs['improvement'], color=colors, alpha=0.8, height=0.6)
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(single_mechs['mech_name'])
    ax.set_xlabel('Improvement over Baseline', fontsize=12)
    ax.set_title('Individual Mechanism Contribution\n(Single Mechanism vs Baseline)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(single_mechs.iterrows()):
        offset = 0.005 if row['improvement'] > 0 else -0.005
        ha = 'left' if row['improvement'] > 0 else 'right'
        ax.text(row['improvement'] + offset, i, f"{row['improvement']:+.3f}", 
               va='center', ha=ha, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mechanism_contribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: mechanism_contribution.png")


def plot_leave_one_out(stats_df: pd.DataFrame, output_dir: str):
    """Analyze leave-one-out ablation results."""
    
    # Extract leave-one-out results
    loo = stats_df[stats_df['condition'].str.startswith('full_minus_')].copy()
    full_bio = stats_df[stats_df['condition'] == 'full_bio']
    
    if loo.empty or full_bio.empty:
        print("  Skipping leave-one-out plot (insufficient data)")
        return
    
    full_bio_acc = full_bio['test_acc_mean'].values[0]
    
    # Calculate drop from full bio
    loo = loo.copy()
    loo['drop'] = full_bio_acc - loo['test_acc_mean']
    loo['mech_removed'] = loo['condition'].str.replace('full_minus_', '')
    
    # Sort by drop (biggest drop = most important mechanism)
    loo = loo.sort_values('drop', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(loo))
    colors = ['#d62728' if drop > 0 else '#2ca02c' for drop in loo['drop']]
    
    bars = ax.barh(y_pos, loo['drop'], color=colors, alpha=0.8, height=0.6)
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(loo['mech_removed'])
    ax.set_xlabel('Performance Drop from Full Bio', fontsize=12)
    ax.set_title('Leave-One-Out Ablation\n(Higher drop = more important mechanism)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(loo.iterrows()):
        offset = 0.002 if row['drop'] > 0 else -0.002
        ha = 'left' if row['drop'] > 0 else 'right'
        ax.text(row['drop'] + offset, i, f"{row['drop']:+.4f}", 
               va='center', ha=ha, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'leave_one_out.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: leave_one_out.png")


def plot_combination_heatmap(stats_df: pd.DataFrame, output_dir: str):
    """Create heatmap of mechanism combinations."""
    
    mechanisms = ['refract', 'lateral', 'homeo', 'stp', 'noise']
    
    # Build presence matrix
    data_rows = []
    for _, row in stats_df.iterrows():
        cond = row['condition']
        presence = [1 if m in cond else 0 for m in mechanisms]
        data_rows.append({
            'condition': cond,
            'test_acc': row['test_acc_mean'],
            **{m: p for m, p in zip(mechanisms, presence)}
        })
    
    df = pd.DataFrame(data_rows)
    
    # Create pivot for heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by accuracy
    df_sorted = df.sort_values('test_acc', ascending=True)
    
    # Create binary matrix for visualization
    mech_matrix = df_sorted[mechanisms].values
    
    # Create heatmap
    im = ax.imshow(mech_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_yticks(np.arange(len(df_sorted)))
    ax.set_yticklabels([f"{row['condition']} ({row['test_acc']:.3f})" 
                       for _, row in df_sorted.iterrows()], fontsize=8)
    ax.set_xticks(np.arange(len(mechanisms)))
    ax.set_xticklabels(mechanisms, fontsize=10)
    ax.set_xlabel('Mechanism', fontsize=12)
    ax.set_title('Mechanism Presence vs Performance', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mechanism Present', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combination_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: combination_heatmap.png")


def generate_summary_report(stats_df: pd.DataFrame, aggregated: Dict, output_dir: str):
    """Generate text summary report."""
    
    report_lines = []
    
    def add(line=""):
        report_lines.append(line)
    
    add("=" * 80)
    add("16-BIT PARITY BIO-CTM ABLATION STUDY REPORT")
    add("=" * 80)
    add()
    
    # Overview
    add("1. OVERVIEW")
    add("-" * 40)
    add(f"Total conditions tested: {len(stats_df)}")
    add(f"Seeds per condition: {stats_df['n_seeds'].iloc[0]}")
    add()
    
    # Top performers
    add("2. TOP PERFORMING CONDITIONS")
    add("-" * 40)
    top5 = stats_df.head(5)
    for _, row in top5.iterrows():
        sig = " ***" if row.get('vs_baseline_significant', False) else ""
        add(f"  {row['condition']}: {row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f}{sig}")
    add()
    
    # Baseline comparison
    add("3. BASELINE COMPARISON")
    add("-" * 40)
    baseline = stats_df[stats_df['condition'] == 'baseline']
    if not baseline.empty:
        bl = baseline.iloc[0]
        add(f"Baseline accuracy: {bl['test_acc_mean']:.4f} ± {bl['test_acc_std']:.4f}")
        add()
        add("Conditions significantly better than baseline (p < 0.05):")
        significant = stats_df[stats_df['vs_baseline_significant'] == True].sort_values('improvement_over_baseline', ascending=False)
        for _, row in significant.iterrows():
            add(f"  {row['condition']}: +{row['improvement_over_baseline']:.4f} (p={row['vs_baseline_p']:.4f}, d={row['cohens_d']:.2f})")
    add()
    
    # Single mechanism analysis
    add("4. SINGLE MECHANISM ANALYSIS")
    add("-" * 40)
    single = stats_df[stats_df['condition'].str.endswith('_only')].sort_values('test_acc_mean', ascending=False)
    if not single.empty:
        for _, row in single.iterrows():
            mech = row['condition'].replace('_only', '')
            imp = row.get('improvement_over_baseline', 0)
            add(f"  {mech}: {row['test_acc_mean']:.4f} (Δ={imp:+.4f})")
    add()
    
    # Best combination analysis
    add("5. COMBINATION ANALYSIS")
    add("-" * 40)
    combos = stats_df[~stats_df['condition'].str.endswith('_only') & 
                      ~stats_df['condition'].isin(['baseline', 'full_bio']) &
                      ~stats_df['condition'].str.startswith('full_minus_')]
    if not combos.empty:
        combos_sorted = combos.sort_values('test_acc_mean', ascending=False).head(5)
        add("Top 5 combinations:")
        for _, row in combos_sorted.iterrows():
            add(f"  {row['condition']}: {row['test_acc_mean']:.4f} ± {row['test_acc_std']:.4f}")
    add()
    
    # Key findings
    add("6. KEY FINDINGS")
    add("-" * 40)
    
    best = stats_df.iloc[0]
    add(f"• Best overall: {best['condition']} ({best['test_acc_mean']:.4f})")
    
    if not single.empty:
        best_single = single.iloc[0]
        add(f"• Best single mechanism: {best_single['condition'].replace('_only', '')} ({best_single['test_acc_mean']:.4f})")
    
    # Check if combining hurts
    full_bio = stats_df[stats_df['condition'] == 'full_bio']
    if not full_bio.empty and not single.empty:
        fb_acc = full_bio.iloc[0]['test_acc_mean']
        best_single_acc = single.iloc[0]['test_acc_mean']
        if fb_acc < best_single_acc:
            add(f"• ⚠️ Full bio ({fb_acc:.4f}) < best single mechanism ({best_single_acc:.4f})")
            add("  → Mechanism interference detected")
        else:
            add(f"• Full bio ({fb_acc:.4f}) ≥ best single mechanism ({best_single_acc:.4f})")
    
    add()
    add("=" * 80)
    
    # Save report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Also print
    for line in report_lines:
        print(line)
    
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze 16-bit Bio-CTM ablation results')
    parser.add_argument('--output_dir', type=str, default='outputs/bio_ablation_16bit',
                       help='Directory containing experiment results')
    parser.add_argument('--save_csv', action='store_true', help='Save statistics as CSV')
    args = parser.parse_args()
    
    print(f"Loading results from: {args.output_dir}")
    print()
    
    # Load results
    results = load_all_results(args.output_dir)
    if not results:
        print(f"No results found in {args.output_dir}")
        return
    
    print(f"Found {len(results)} experiments")
    
    # Aggregate
    aggregated = aggregate_by_condition(results)
    print(f"Found {len(aggregated)} unique conditions")
    print()
    
    # Compute statistics
    print("Computing statistics...")
    stats_df = compute_statistics(aggregated)
    
    # Save CSV if requested
    if args.save_csv:
        csv_path = os.path.join(args.output_dir, 'statistics.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"  Saved: statistics.csv")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_learning_curves(aggregated, args.output_dir)
    plot_bar_comparison(stats_df, aggregated, args.output_dir)
    plot_mechanism_contribution(stats_df, args.output_dir)
    plot_leave_one_out(stats_df, args.output_dir)
    plot_combination_heatmap(stats_df, args.output_dir)
    
    # Generate report
    print("\n" + "=" * 80)
    generate_summary_report(stats_df, aggregated, args.output_dir)


if __name__ == '__main__':
    main()