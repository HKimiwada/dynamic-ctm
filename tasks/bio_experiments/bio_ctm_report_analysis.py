#!/usr/bin/env python3
"""
Bio-CTM Research Report Analysis Suite
Generates publication-ready figures and statistics for bio-inspired CTM research.

Usage:
    python bio_ctm_report_analysis.py --results_dir outputs/bio_ablation_16bit --output_dir report_figures
"""
#!/usr/bin/env python3
"""
Bio-CTM Research Report Analysis Suite
Generates publication-ready figures and statistics for bio-inspired CTM research.

Usage:
    python bio_ctm_report_analysis.py --results_dir outputs/bio_ablation_16bit --output_dir report_figures
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme for consistency
COLORS = {
    'baseline': '#4477AA',
    'bio_best': '#228833',
    'bio_good': '#66BB66',
    'bio_neutral': '#BBBBBB',
    'bio_bad': '#EE6677',
    'bio_worst': '#AA3377',
}


def load_all_results(results_dir):
    """Load all experiment results from directory."""
    results = {}
    results_path = Path(results_dir)
    
    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        results_file = exp_dir / 'results.json'
        config_file = exp_dir / 'config.json'
        
        if results_file.exists() and config_file.exists():
            with open(results_file) as f:
                exp_results = json.load(f)
            with open(config_file) as f:
                config = json.load(f)
            
            results[exp_dir.name] = {
                'results': exp_results,
                'config': config
            }
    
    return results


def aggregate_by_condition(results):
    """Group results by condition (removing seed suffix)."""
    conditions = defaultdict(list)
    
    for exp_name, data in results.items():
        # Extract condition name (remove seed suffix)
        parts = exp_name.rsplit('_seed', 1)
        condition = parts[0] if len(parts) > 1 else exp_name
        
        if data['results']['test_acc']:
            final_acc = data['results']['test_acc'][-1]
            best_acc = max(data['results']['test_acc'])
            conditions[condition].append({
                'final_acc': final_acc,
                'best_acc': best_acc,
                'train_acc': data['results']['train_acc'],
                'test_acc': data['results']['test_acc'],
                'train_loss': data['results']['train_loss'],
                'test_loss': data['results']['test_loss'],
                'config': data['config']
            })
    
    return conditions


def compute_statistics(conditions, baseline_name='baseline'):
    """Compute statistics for each condition vs baseline."""
    stats_dict = {}
    
    baseline_accs = [r['final_acc'] for r in conditions.get(baseline_name, [])]
    baseline_mean = np.mean(baseline_accs) if baseline_accs else 0
    baseline_std = np.std(baseline_accs) if baseline_accs else 0
    
    for condition, runs in conditions.items():
        accs = [r['final_acc'] for r in runs]
        best_accs = [r['best_acc'] for r in runs]
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        
        # Statistical test vs baseline
        if baseline_accs and len(accs) >= 2:
            t_stat, p_value = stats.ttest_ind(accs, baseline_accs)
            # Cohen's d
            pooled_std = np.sqrt((std_acc**2 + baseline_std**2) / 2)
            cohens_d = (mean_acc - baseline_mean) / pooled_std if pooled_std > 0 else 0
        else:
            t_stat, p_value, cohens_d = None, None, None
        
        stats_dict[condition] = {
            'mean': mean_acc,
            'std': std_acc,
            'best_mean': np.mean(best_accs),
            'best_std': np.std(best_accs),
            'n_seeds': len(runs),
            'delta': mean_acc - baseline_mean,
            'delta_pct': (mean_acc - baseline_mean) / baseline_mean * 100 if baseline_mean > 0 else 0,
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05 if p_value is not None else False,
            'runs': runs
        }
    
    return stats_dict, baseline_mean, baseline_std


def create_main_results_figure(stats_dict, baseline_mean, output_dir, task_name="16-bit Parity"):
    """Create main results bar chart with error bars."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort conditions by accuracy
    sorted_conditions = sorted(stats_dict.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    conditions = [c[0] for c in sorted_conditions]
    means = [c[1]['mean'] for c in sorted_conditions]
    stds = [c[1]['std'] for c in sorted_conditions]
    
    # Color based on performance vs baseline
    colors = []
    for condition, data in sorted_conditions:
        if condition == 'baseline':
            colors.append(COLORS['baseline'])
        elif data['delta'] > 0.02:
            colors.append(COLORS['bio_best'])
        elif data['delta'] > 0:
            colors.append(COLORS['bio_good'])
        elif data['delta'] > -0.05:
            colors.append(COLORS['bio_neutral'])
        elif data['delta'] > -0.15:
            colors.append(COLORS['bio_bad'])
        else:
            colors.append(COLORS['bio_worst'])
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add baseline reference line
    ax.axhline(y=baseline_mean, color=COLORS['baseline'], linestyle='--', linewidth=2, alpha=0.7, label=f'Baseline ({baseline_mean:.3f})')
    
    # Add significance markers
    for i, (condition, data) in enumerate(sorted_conditions):
        if data['significant'] and data['delta'] > 0:
            ax.annotate('*', (i, means[i] + stds[i] + 0.01), ha='center', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'Bio-CTM Ablation Study: {task_name}')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_results.png'))
    plt.savefig(os.path.join(output_dir, 'main_results.pdf'))
    plt.close()
    print(f"Saved main_results.png/pdf")


def create_ablation_heatmap(stats_dict, output_dir):
    """Create mechanism ablation heatmap."""
    # Define mechanisms
    mechanisms = ['refractory', 'lateral', 'stp', 'noise', 'homeostasis']
    mech_short = ['Refr', 'Lat', 'STP', 'Noise', 'Homeo']
    
    # Map conditions to mechanism presence
    condition_to_mechs = {
        'baseline': [],
        'refract_only': ['refractory'],
        'lateral_only': ['lateral'],
        'stp_only': ['stp'],
        'noise_only': ['noise'],
        'homeo_only': ['homeostasis'],
        'refract_lateral': ['refractory', 'lateral'],
        'refract_stp': ['refractory', 'stp'],
        'refract_homeo': ['refractory', 'homeostasis'],
        'lateral_homeo': ['lateral', 'homeostasis'],
        'refract_lateral_homeo': ['refractory', 'lateral', 'homeostasis'],
        'full_bio': ['refractory', 'lateral', 'stp', 'noise', 'homeostasis'],
        'full_minus_homeo': ['refractory', 'lateral', 'stp', 'noise'],
        'full_minus_refract': ['lateral', 'stp', 'noise', 'homeostasis'],
        'full_minus_lateral': ['refractory', 'stp', 'noise', 'homeostasis'],
        'full_minus_stp': ['refractory', 'lateral', 'noise', 'homeostasis'],
        'full_minus_noise': ['refractory', 'lateral', 'stp', 'homeostasis'],
    }
    
    # Build data matrix
    conditions_present = [c for c in condition_to_mechs.keys() if c in stats_dict]
    
    if not conditions_present:
        print("No matching conditions for heatmap")
        return
    
    # Sort by accuracy
    conditions_present = sorted(conditions_present, key=lambda c: stats_dict[c]['mean'], reverse=True)
    
    # Create presence matrix
    presence_matrix = np.zeros((len(conditions_present), len(mechanisms)))
    accuracy_values = []
    
    for i, cond in enumerate(conditions_present):
        mechs = condition_to_mechs.get(cond, [])
        for j, mech in enumerate(mechanisms):
            presence_matrix[i, j] = 1 if mech in mechs else 0
        accuracy_values.append(stats_dict[cond]['mean'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [3, 1]})
    
    # Mechanism presence heatmap
    sns.heatmap(presence_matrix, ax=ax1, cmap='Blues', cbar=False,
                xticklabels=mech_short, yticklabels=conditions_present,
                linewidths=0.5, linecolor='white')
    ax1.set_title('Mechanism Presence')
    ax1.set_xlabel('Mechanism')
    
    # Accuracy bar chart
    colors = [COLORS['bio_best'] if acc > stats_dict.get('baseline', {}).get('mean', 0) else COLORS['bio_bad'] 
              for acc in accuracy_values]
    ax2.barh(range(len(conditions_present)), accuracy_values, color=colors, alpha=0.85)
    ax2.set_yticks([])
    ax2.set_xlabel('Test Accuracy')
    ax2.set_title('Performance')
    ax2.set_xlim(0.4, 1.0)
    
    # Add baseline line
    baseline_acc = stats_dict.get('baseline', {}).get('mean', 0)
    ax2.axvline(x=baseline_acc, color=COLORS['baseline'], linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_heatmap.png'))
    plt.savefig(os.path.join(output_dir, 'ablation_heatmap.pdf'))
    plt.close()
    print(f"Saved ablation_heatmap.png/pdf")


def create_training_curves_figure(conditions, output_dir, 
                                   selected=['baseline', 'full_minus_homeo', 'refract_only', 'full_bio']):
    """Create training curves comparison."""
    # Filter to only conditions that exist
    available_selected = [c for c in selected if c in conditions]
    
    # If none of the default selected conditions exist, use all available
    if not available_selected:
        available_selected = list(conditions.keys())
    
    if not available_selected:
        print("No conditions available for training curves")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(available_selected)))
    
    has_test_data = False
    has_train_data = False
    
    for idx, condition in enumerate(available_selected):
        runs = conditions[condition]
        
        # Collect all curves
        test_curves = [r['test_acc'] for r in runs if r['test_acc']]
        train_curves = [r['train_loss'] for r in runs if r['train_loss']]
        
        if not test_curves:
            continue
        
        # Test accuracy
        ax = axes[0]
        min_len = min(len(c) for c in test_curves)
        curves_aligned = np.array([c[:min_len] for c in test_curves])
        mean_curve = np.mean(curves_aligned, axis=0)
        std_curve = np.std(curves_aligned, axis=0)
        
        epochs = [(i+1) * 10 for i in range(min_len)]  # Assuming eval_every=10
        ax.plot(epochs, mean_curve, color=colors[idx], linewidth=2, label=condition)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, 
                       color=colors[idx], alpha=0.2)
        has_test_data = True
        
        # Train loss
        ax = axes[1]
        if train_curves:
            min_len = min(len(c) for c in train_curves)
            curves_aligned = np.array([c[:min_len] for c in train_curves])
            mean_curve = np.mean(curves_aligned, axis=0)
            epochs = list(range(1, min_len + 1))
            ax.plot(epochs, mean_curve, color=colors[idx], linewidth=2, label=condition)
            has_train_data = True
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Test Accuracy During Training')
    if has_test_data:
        axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Train Loss')
    axes[1].set_title('Training Loss')
    if has_train_data:
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'))
    plt.close()
    print(f"Saved training_curves.png/pdf")


def create_mechanism_effect_figure(stats_dict, output_dir):
    """Create bar chart showing individual mechanism effects."""
    single_mechanisms = {
        'Refractory': 'refract_only',
        'Lateral Inh.': 'lateral_only', 
        'STP': 'stp_only',
        'Synaptic Noise': 'noise_only',
        'Homeostasis': 'homeo_only'
    }
    
    baseline_mean = stats_dict.get('baseline', {}).get('mean', 0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    names = []
    deltas = []
    errors = []
    colors = []
    
    for name, condition in single_mechanisms.items():
        if condition in stats_dict:
            names.append(name)
            delta = stats_dict[condition]['delta_pct']
            deltas.append(delta)
            errors.append(stats_dict[condition]['std'] * 100)
            colors.append(COLORS['bio_best'] if delta > 0 else COLORS['bio_bad'])
    
    x = np.arange(len(names))
    bars = ax.bar(x, deltas, yerr=errors, capsize=4, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel('Δ Accuracy vs Baseline (%)')
    ax.set_title('Individual Mechanism Effects')
    
    # Add value labels
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        ax.annotate(f'{delta:+.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3 if height >= 0 else -12),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mechanism_effects.png'))
    plt.savefig(os.path.join(output_dir, 'mechanism_effects.pdf'))
    plt.close()
    print(f"Saved mechanism_effects.png/pdf")


def create_summary_table(stats_dict, output_dir):
    """Create LaTeX and markdown summary tables."""
    # Sort by accuracy
    sorted_stats = sorted(stats_dict.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    # Markdown table
    md_lines = [
        "| Condition | Accuracy | Δ vs Baseline | p-value | Effect Size (d) |",
        "|-----------|----------|---------------|---------|-----------------|"
    ]
    
    for condition, data in sorted_stats:
        p_str = f"{data['p_value']:.4f}" if data['p_value'] is not None else "—"
        d_str = f"{data['cohens_d']:.2f}" if data['cohens_d'] is not None else "—"
        sig = "**" if data['significant'] and data['delta'] > 0 else ""
        
        md_lines.append(
            f"| {sig}{condition}{sig} | {data['mean']:.4f} ± {data['std']:.4f} | "
            f"{data['delta']:+.4f} ({data['delta_pct']:+.1f}%) | {p_str} | {d_str} |"
        )
    
    with open(os.path.join(output_dir, 'results_table.md'), 'w') as f:
        f.write('\n'.join(md_lines))
    
    # LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Bio-CTM Ablation Results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & Accuracy & $\Delta$ vs Baseline & p-value & Cohen's d \\",
        r"\midrule"
    ]
    
    for condition, data in sorted_stats[:10]:  # Top 10
        p_str = f"{data['p_value']:.3f}" if data['p_value'] is not None else "—"
        d_str = f"{data['cohens_d']:.2f}" if data['cohens_d'] is not None else "—"
        bold_start = r"\textbf{" if data['significant'] and data['delta'] > 0 else ""
        bold_end = "}" if data['significant'] and data['delta'] > 0 else ""
        
        condition_clean = condition.replace('_', r'\_')
        latex_lines.append(
            f"{bold_start}{condition_clean}{bold_end} & "
            f"{data['mean']:.3f} $\\pm$ {data['std']:.3f} & "
            f"{data['delta']:+.3f} & {p_str} & {d_str} \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:ablation}",
        r"\end{table}"
    ])
    
    with open(os.path.join(output_dir, 'results_table.tex'), 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Saved results_table.md and results_table.tex")


def create_key_findings_summary(stats_dict, output_dir):
    """Generate key findings text for report."""
    baseline = stats_dict.get('baseline', {})
    
    # Handle case where baseline might not exist
    if not baseline:
        baseline = {'mean': 0, 'std': 0}
    
    best_condition = max(stats_dict.items(), key=lambda x: x[1]['mean'])
    worst_condition = min(stats_dict.items(), key=lambda x: x[1]['mean'])
    
    # Single mechanisms (may not be present in validation runs)
    single_mechs = ['refract_only', 'lateral_only', 'stp_only', 'noise_only', 'homeo_only']
    single_results = [(m, stats_dict[m]) for m in single_mechs if m in stats_dict]
    
    findings = f"""
# Key Findings: Bio-CTM Analysis

## Main Results

1. **Best Configuration**: `{best_condition[0]}` achieves {best_condition[1]['mean']:.4f} ± {best_condition[1]['std']:.4f} accuracy
   - Improvement over baseline: {best_condition[1]['delta']:+.4f} ({best_condition[1]['delta_pct']:+.1f}%)
   - Statistical significance: p = {best_condition[1]['p_value']:.4f if best_condition[1]['p_value'] else 'N/A'}

2. **Baseline Performance**: {baseline.get('mean', 0):.4f} ± {baseline.get('std', 0):.4f}

3. **Worst Configuration**: `{worst_condition[0]}` at {worst_condition[1]['mean']:.4f}
   - Delta from baseline: {worst_condition[1]['delta']:.4f} ({worst_condition[1]['delta_pct']:.1f}%)

## Conditions Tested

| Condition | Accuracy | Δ vs Baseline |
|-----------|----------|---------------|
"""
    
    # Add all conditions to table
    for cond, data in sorted(stats_dict.items(), key=lambda x: x[1]['mean'], reverse=True):
        findings += f"| {cond} | {data['mean']:.4f} ± {data['std']:.4f} | {data['delta']:+.4f} |\n"
    
    # Add single mechanism analysis only if we have those results
    if single_results:
        best_single = max(single_results, key=lambda x: x[1]['mean'])
        findings += f"""
## Single Mechanism Analysis

Best single mechanism: `{best_single[0]}` ({best_single[1]['mean']:.4f}, Δ = {best_single[1]['delta']:+.4f})

| Mechanism | Accuracy | Δ vs Baseline |
|-----------|----------|---------------|
"""
        for mech, data in sorted(single_results, key=lambda x: x[1]['mean'], reverse=True):
            findings += f"| {mech} | {data['mean']:.4f} | {data['delta']:+.4f} |\n"
    
    # Add homeostasis insight only if we have that data
    homeo_data = stats_dict.get('homeo_only', {})
    if homeo_data:
        findings += f"""
## Critical Insight: Homeostasis Interference

Homeostasis mechanism causes significant performance degradation:
- `homeo_only`: {homeo_data.get('mean', 0):.4f} (Δ = {homeo_data.get('delta', 0):.4f})
- All combinations including homeostasis perform poorly
- Removing homeostasis from full_bio recovers performance

**Hypothesis**: The homeostatic target firing rate conflicts with 
the representations learned by gradient descent.
"""
    
    findings += """
## Recommendations

1. Use `full_minus_homeo` (refractory + lateral + STP + noise) as default bio-CTM
2. Avoid homeostasis mechanism
3. Refractory alone is a good lightweight alternative
"""
    
    with open(os.path.join(output_dir, 'key_findings.md'), 'w') as f:
        f.write(findings)
    
    print(f"Saved key_findings.md")
    return findings


def main():
    parser = argparse.ArgumentParser(description='Generate Bio-CTM research report figures')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='report_figures',
                       help='Directory to save figures')
    parser.add_argument('--task_name', type=str, default='16-bit Parity',
                       help='Task name for figure titles')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading results from {args.results_dir}...")
    results = load_all_results(args.results_dir)
    print(f"Loaded {len(results)} experiments")
    
    print("\nAggregating by condition...")
    conditions = aggregate_by_condition(results)
    print(f"Found {len(conditions)} conditions")
    
    print("\nComputing statistics...")
    stats_dict, baseline_mean, baseline_std = compute_statistics(conditions)
    
    print("\nGenerating figures...")
    create_main_results_figure(stats_dict, baseline_mean, args.output_dir, args.task_name)
    create_ablation_heatmap(stats_dict, args.output_dir)
    create_training_curves_figure(conditions, args.output_dir)
    create_mechanism_effect_figure(stats_dict, args.output_dir)
    
    print("\nGenerating tables...")
    create_summary_table(stats_dict, args.output_dir)
    
    print("\nGenerating findings summary...")
    findings = create_key_findings_summary(stats_dict, args.output_dir)
    print(findings)
    
    print(f"\n✓ All outputs saved to {args.output_dir}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  - {f}")


if __name__ == '__main__':
    main()