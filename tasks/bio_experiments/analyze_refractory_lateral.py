# tasks/bio_experiments/analyze_refractory_lateral.py
"""
Comprehensive analysis comparing Refractory + Lateral Inhibition vs Baseline CTM.
Extracts training dynamics, convergence patterns, and statistical significance.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import argparse


def load_experiment(exp_dir, exp_name):
    """Load results and config for a single experiment."""
    exp_path = os.path.join(exp_dir, exp_name)
    
    results_file = os.path.join(exp_path, 'results.json')
    config_file = os.path.join(exp_path, 'config.json')
    
    if not os.path.exists(results_file) or not os.path.exists(config_file):
        print(f"Warning: Missing files for {exp_name}")
        return None
    
    with open(results_file) as f:
        results = json.load(f)
    with open(config_file) as f:
        config = json.load(f)
    
    return {'results': results, 'config': config, 'name': exp_name}


def load_all_experiments(baseline_dir, bio_dir, seeds):
    """Load all baseline and bio experiments."""
    experiments = {
        'baseline': {},
        'refractory_lateral': {}
    }
    
    for seed in seeds:
        # Load baseline
        baseline_name = f'baseline_seed{seed}'
        baseline_data = load_experiment(baseline_dir, baseline_name)
        if baseline_data:
            experiments['baseline'][seed] = baseline_data
        
        # Load refractory + lateral
        bio_name = f'seed{seed}_refractory_plus_lateral'
        bio_data = load_experiment(bio_dir, bio_name)
        if bio_data:
            experiments['refractory_lateral'][seed] = bio_data
    
    return experiments


def compute_summary_statistics(experiments):
    """Compute summary statistics for each condition."""
    summary = {}
    
    for condition, seed_data in experiments.items():
        if not seed_data:
            continue
            
        # Extract final/best metrics
        final_test_accs = []
        best_test_accs = []
        final_train_accs = []
        best_train_accs = []
        final_test_losses = []
        final_train_losses = []
        
        for seed, data in seed_data.items():
            results = data['results']
            
            if results['test_acc']:
                final_test_accs.append(results['test_acc'][-1])
                best_test_accs.append(max(results['test_acc']))
            if results['train_acc']:
                final_train_accs.append(results['train_acc'][-1])
                best_train_accs.append(max(results['train_acc']))
            if results['test_loss']:
                final_test_losses.append(results['test_loss'][-1])
            if results['train_loss']:
                final_train_losses.append(results['train_loss'][-1])
        
        summary[condition] = {
            'final_test_acc': {
                'mean': np.mean(final_test_accs),
                'std': np.std(final_test_accs),
                'values': final_test_accs
            },
            'best_test_acc': {
                'mean': np.mean(best_test_accs),
                'std': np.std(best_test_accs),
                'values': best_test_accs
            },
            'final_train_acc': {
                'mean': np.mean(final_train_accs),
                'std': np.std(final_train_accs),
                'values': final_train_accs
            },
            'best_train_acc': {
                'mean': np.mean(best_train_accs),
                'std': np.std(best_train_accs),
                'values': best_train_accs
            },
            'final_test_loss': {
                'mean': np.mean(final_test_losses),
                'std': np.std(final_test_losses),
                'values': final_test_losses
            },
            'final_train_loss': {
                'mean': np.mean(final_train_losses),
                'std': np.std(final_train_losses),
                'values': final_train_losses
            },
            'n_seeds': len(seed_data)
        }
    
    return summary


def compute_statistical_tests(summary):
    """Perform statistical tests comparing conditions."""
    if 'baseline' not in summary or 'refractory_lateral' not in summary:
        return None
    
    baseline = summary['baseline']
    bio = summary['refractory_lateral']
    
    tests = {}
    
    # Paired t-test on final test accuracy
    if len(baseline['final_test_acc']['values']) >= 2:
        t_stat, p_value = stats.ttest_ind(
            bio['final_test_acc']['values'],
            baseline['final_test_acc']['values']
        )
        tests['final_test_acc_ttest'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_0.05': p_value < 0.05,
            'significant_0.10': p_value < 0.10
        }
    
    # Paired t-test on best test accuracy
    if len(baseline['best_test_acc']['values']) >= 2:
        t_stat, p_value = stats.ttest_ind(
            bio['best_test_acc']['values'],
            baseline['best_test_acc']['values']
        )
        tests['best_test_acc_ttest'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_0.05': p_value < 0.05,
            'significant_0.10': p_value < 0.10
        }
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (baseline['final_test_acc']['std']**2 + bio['final_test_acc']['std']**2) / 2
    )
    if pooled_std > 0:
        cohens_d = (bio['final_test_acc']['mean'] - baseline['final_test_acc']['mean']) / pooled_std
        tests['cohens_d'] = cohens_d
    
    # Improvement metrics
    tests['absolute_improvement'] = bio['final_test_acc']['mean'] - baseline['final_test_acc']['mean']
    tests['relative_improvement'] = (
        (bio['final_test_acc']['mean'] - baseline['final_test_acc']['mean']) 
        / baseline['final_test_acc']['mean'] * 100
    )
    
    return tests


def analyze_training_dynamics(experiments):
    """Analyze training dynamics: convergence speed, stability, etc."""
    dynamics = {}
    
    for condition, seed_data in experiments.items():
        if not seed_data:
            continue
        
        condition_dynamics = {
            'convergence_epoch': [],
            'early_accuracy': [],  # Accuracy at epoch 20
            'mid_accuracy': [],    # Accuracy at epoch 50
            'accuracy_variance_over_training': [],
            'loss_smoothness': []
        }
        
        for seed, data in seed_data.items():
            results = data['results']
            test_accs = results['test_acc']
            train_losses = results['train_loss']
            
            if not test_accs:
                continue
            
            # Convergence: first epoch where accuracy > 95% of final
            final_acc = test_accs[-1]
            threshold = 0.95 * final_acc
            convergence_epoch = len(test_accs) * 10  # Default to end (eval_every=10)
            for i, acc in enumerate(test_accs):
                if acc >= threshold:
                    convergence_epoch = (i + 1) * 10
                    break
            condition_dynamics['convergence_epoch'].append(convergence_epoch)
            
            # Early and mid accuracy
            if len(test_accs) >= 2:
                condition_dynamics['early_accuracy'].append(test_accs[1])  # Epoch 20
            if len(test_accs) >= 5:
                condition_dynamics['mid_accuracy'].append(test_accs[4])    # Epoch 50
            
            # Variance of accuracy over last half of training
            if len(test_accs) >= 4:
                second_half = test_accs[len(test_accs)//2:]
                condition_dynamics['accuracy_variance_over_training'].append(np.var(second_half))
            
            # Loss smoothness: average absolute difference between consecutive losses
            if len(train_losses) >= 2:
                diffs = np.abs(np.diff(train_losses))
                condition_dynamics['loss_smoothness'].append(np.mean(diffs))
        
        # Aggregate
        dynamics[condition] = {}
        for metric, values in condition_dynamics.items():
            if values:
                dynamics[condition][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
    
    return dynamics


def analyze_per_seed_comparison(experiments):
    """Direct per-seed comparison."""
    seeds = set(experiments['baseline'].keys()) & set(experiments['refractory_lateral'].keys())
    
    comparisons = []
    for seed in sorted(seeds):
        baseline = experiments['baseline'][seed]['results']
        bio = experiments['refractory_lateral'][seed]['results']
        
        comparison = {
            'seed': seed,
            'baseline_final_test_acc': baseline['test_acc'][-1] if baseline['test_acc'] else None,
            'bio_final_test_acc': bio['test_acc'][-1] if bio['test_acc'] else None,
            'baseline_best_test_acc': max(baseline['test_acc']) if baseline['test_acc'] else None,
            'bio_best_test_acc': max(bio['test_acc']) if bio['test_acc'] else None,
            'baseline_final_train_loss': baseline['train_loss'][-1] if baseline['train_loss'] else None,
            'bio_final_train_loss': bio['train_loss'][-1] if bio['train_loss'] else None,
        }
        
        if comparison['baseline_final_test_acc'] and comparison['bio_final_test_acc']:
            comparison['improvement'] = comparison['bio_final_test_acc'] - comparison['baseline_final_test_acc']
            comparison['bio_wins'] = comparison['bio_final_test_acc'] > comparison['baseline_final_test_acc']
        
        comparisons.append(comparison)
    
    return comparisons


def plot_training_curves(experiments, output_dir):
    """Plot training curves for both conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'baseline': 'tab:blue', 'refractory_lateral': 'tab:orange'}
    labels = {'baseline': 'Baseline CTM', 'refractory_lateral': 'Refractory + Lateral'}
    
    # Test Accuracy
    ax = axes[0, 0]
    for condition, seed_data in experiments.items():
        all_curves = []
        for seed, data in seed_data.items():
            test_acc = data['results']['test_acc']
            if test_acc:
                epochs = [(i+1) * 10 for i in range(len(test_acc))]
                ax.plot(epochs, test_acc, color=colors[condition], alpha=0.3, linewidth=1)
                all_curves.append(test_acc)
        
        # Plot mean
        if all_curves:
            min_len = min(len(c) for c in all_curves)
            mean_curve = np.mean([c[:min_len] for c in all_curves], axis=0)
            std_curve = np.std([c[:min_len] for c in all_curves], axis=0)
            epochs = [(i+1) * 10 for i in range(min_len)]
            ax.plot(epochs, mean_curve, color=colors[condition], linewidth=2, label=labels[condition])
            ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, 
                          color=colors[condition], alpha=0.2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Train Accuracy
    ax = axes[0, 1]
    for condition, seed_data in experiments.items():
        all_curves = []
        for seed, data in seed_data.items():
            train_acc = data['results']['train_acc']
            if train_acc:
                epochs = list(range(1, len(train_acc) + 1))
                ax.plot(epochs, train_acc, color=colors[condition], alpha=0.3, linewidth=0.5)
                all_curves.append(train_acc)
        
        if all_curves:
            min_len = min(len(c) for c in all_curves)
            mean_curve = np.mean([c[:min_len] for c in all_curves], axis=0)
            epochs = list(range(1, min_len + 1))
            ax.plot(epochs, mean_curve, color=colors[condition], linewidth=2, label=labels[condition])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Accuracy')
    ax.set_title('Train Accuracy Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test Loss
    ax = axes[1, 0]
    for condition, seed_data in experiments.items():
        all_curves = []
        for seed, data in seed_data.items():
            test_loss = data['results']['test_loss']
            if test_loss:
                epochs = [(i+1) * 10 for i in range(len(test_loss))]
                ax.plot(epochs, test_loss, color=colors[condition], alpha=0.3, linewidth=1)
                all_curves.append(test_loss)
        
        if all_curves:
            min_len = min(len(c) for c in all_curves)
            mean_curve = np.mean([c[:min_len] for c in all_curves], axis=0)
            std_curve = np.std([c[:min_len] for c in all_curves], axis=0)
            epochs = [(i+1) * 10 for i in range(min_len)]
            ax.plot(epochs, mean_curve, color=colors[condition], linewidth=2, label=labels[condition])
            ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve,
                          color=colors[condition], alpha=0.2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Train Loss
    ax = axes[1, 1]
    for condition, seed_data in experiments.items():
        all_curves = []
        for seed, data in seed_data.items():
            train_loss = data['results']['train_loss']
            if train_loss:
                epochs = list(range(1, len(train_loss) + 1))
                ax.plot(epochs, train_loss, color=colors[condition], alpha=0.3, linewidth=0.5)
                all_curves.append(train_loss)
        
        if all_curves:
            min_len = min(len(c) for c in all_curves)
            mean_curve = np.mean([c[:min_len] for c in all_curves], axis=0)
            epochs = list(range(1, min_len + 1))
            ax.plot(epochs, mean_curve, color=colors[condition], linewidth=2, label=labels[condition])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title('Train Loss Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Saved training curves to {output_dir}/training_curves.png")


def plot_per_seed_comparison(comparisons, output_dir):
    """Bar plot comparing each seed."""
    seeds = [c['seed'] for c in comparisons]
    baseline_accs = [c['baseline_final_test_acc'] for c in comparisons]
    bio_accs = [c['bio_final_test_acc'] for c in comparisons]
    
    x = np.arange(len(seeds))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Final test accuracy comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline CTM', color='tab:blue', alpha=0.8)
    bars2 = ax.bar(x + width/2, bio_accs, width, label='Refractory + Lateral', color='tab:orange', alpha=0.8)
    
    ax.set_xlabel('Seed')
    ax.set_ylabel('Final Test Accuracy')
    ax.set_title('Per-Seed Final Test Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # Improvement plot
    ax = axes[1]
    improvements = [c['improvement'] for c in comparisons if 'improvement' in c]
    colors = ['tab:green' if imp > 0 else 'tab:red' for imp in improvements]
    bars = ax.bar(x, improvements, color=colors, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Accuracy Improvement (Bio - Baseline)')
    ax.set_title('Per-Seed Improvement from Bio-Inspired Mechanisms')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(f'{imp:+.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3 if height >= 0 else -12), textcoords="offset points", 
                   ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_seed_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved per-seed comparison to {output_dir}/per_seed_comparison.png")


def plot_summary_statistics(summary, output_dir):
    """Plot summary statistics with error bars."""
    conditions = list(summary.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    colors = {'baseline': 'tab:blue', 'refractory_lateral': 'tab:orange'}
    labels = {'baseline': 'Baseline CTM', 'refractory_lateral': 'Refractory + Lateral'}
    
    # Final Test Accuracy
    ax = axes[0]
    x = np.arange(len(conditions))
    means = [summary[c]['final_test_acc']['mean'] for c in conditions]
    stds = [summary[c]['final_test_acc']['std'] for c in conditions]
    bars = ax.bar(x, means, yerr=stds, capsize=5, 
                  color=[colors[c] for c in conditions], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in conditions])
    ax.set_ylabel('Final Test Accuracy')
    ax.set_title('Final Test Accuracy')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f'{mean:.4f}±{std:.4f}', 
                   xy=(bar.get_x() + bar.get_width()/2, mean + std),
                   xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10)
    
    # Best Test Accuracy
    ax = axes[1]
    means = [summary[c]['best_test_acc']['mean'] for c in conditions]
    stds = [summary[c]['best_test_acc']['std'] for c in conditions]
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors[c] for c in conditions], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in conditions])
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Best Test Accuracy (Peak Performance)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f'{mean:.4f}±{std:.4f}',
                   xy=(bar.get_x() + bar.get_width()/2, mean + std),
                   xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10)
    
    # Final Train Loss
    ax = axes[2]
    means = [summary[c]['final_train_loss']['mean'] for c in conditions]
    stds = [summary[c]['final_train_loss']['std'] for c in conditions]
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors[c] for c in conditions], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in conditions])
    ax.set_ylabel('Final Train Loss')
    ax.set_title('Final Training Loss')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f'{mean:.4f}±{std:.4f}',
                   xy=(bar.get_x() + bar.get_width()/2, mean + std),
                   xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=150)
    plt.close()
    print(f"Saved summary statistics to {output_dir}/summary_statistics.png")


def plot_training_dynamics(dynamics, output_dir):
    """Plot training dynamics comparison."""
    if len(dynamics) < 2:
        print("Not enough conditions to plot dynamics comparison")
        return
    
    conditions = list(dynamics.keys())
    colors = {'baseline': 'tab:blue', 'refractory_lateral': 'tab:orange'}
    labels = {'baseline': 'Baseline CTM', 'refractory_lateral': 'Refractory + Lateral'}
    
    metrics = ['convergence_epoch', 'early_accuracy', 'mid_accuracy', 'accuracy_variance_over_training']
    metric_labels = ['Convergence Epoch\n(95% of final)', 'Early Accuracy\n(Epoch 20)', 
                     'Mid Accuracy\n(Epoch 50)', 'Accuracy Variance\n(2nd half of training)']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        x = np.arange(len(conditions))
        means = []
        stds = []
        for c in conditions:
            if metric in dynamics[c]:
                means.append(dynamics[c][metric]['mean'])
                stds.append(dynamics[c][metric]['std'])
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[colors[c] for c in conditions], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in conditions], rotation=15, ha='right')
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_dynamics.png'), dpi=150)
    plt.close()
    print(f"Saved training dynamics to {output_dir}/training_dynamics.png")


def print_full_report(summary, tests, dynamics, comparisons, output_dir):
    """Print comprehensive text report and save to file."""
    report_lines = []
    
    def add_line(line=""):
        report_lines.append(line)
        print(line)
    
    add_line("=" * 80)
    add_line("COMPREHENSIVE ANALYSIS: Refractory + Lateral Inhibition vs Baseline CTM")
    add_line("=" * 80)
    add_line()
    
    # Summary Statistics
    add_line("1. SUMMARY STATISTICS")
    add_line("-" * 40)
    add_line(f"{'Metric':<30} {'Baseline':<20} {'Bio-Inspired':<20}")
    add_line("-" * 70)
    
    if 'baseline' in summary and 'refractory_lateral' in summary:
        base = summary['baseline']
        bio = summary['refractory_lateral']
        
        add_line(f"{'Final Test Accuracy':<30} "
                f"{base['final_test_acc']['mean']:.4f} ± {base['final_test_acc']['std']:.4f}    "
                f"{bio['final_test_acc']['mean']:.4f} ± {bio['final_test_acc']['std']:.4f}")
        add_line(f"{'Best Test Accuracy':<30} "
                f"{base['best_test_acc']['mean']:.4f} ± {base['best_test_acc']['std']:.4f}    "
                f"{bio['best_test_acc']['mean']:.4f} ± {bio['best_test_acc']['std']:.4f}")
        add_line(f"{'Final Train Accuracy':<30} "
                f"{base['final_train_acc']['mean']:.4f} ± {base['final_train_acc']['std']:.4f}    "
                f"{bio['final_train_acc']['mean']:.4f} ± {bio['final_train_acc']['std']:.4f}")
        add_line(f"{'Final Train Loss':<30} "
                f"{base['final_train_loss']['mean']:.4f} ± {base['final_train_loss']['std']:.4f}    "
                f"{bio['final_train_loss']['mean']:.4f} ± {bio['final_train_loss']['std']:.4f}")
        add_line(f"{'Number of Seeds':<30} {base['n_seeds']:<20} {bio['n_seeds']:<20}")
    
    add_line()
    
    # Statistical Tests
    add_line("2. STATISTICAL TESTS")
    add_line("-" * 40)
    
    if tests:
        if 'final_test_acc_ttest' in tests:
            t = tests['final_test_acc_ttest']
            add_line(f"Final Test Accuracy t-test:")
            add_line(f"  t-statistic: {t['t_statistic']:.4f}")
            add_line(f"  p-value: {t['p_value']:.4f}")
            add_line(f"  Significant at α=0.05: {t['significant_0.05']}")
            add_line(f"  Significant at α=0.10: {t['significant_0.10']}")
        
        if 'best_test_acc_ttest' in tests:
            t = tests['best_test_acc_ttest']
            add_line(f"\nBest Test Accuracy t-test:")
            add_line(f"  t-statistic: {t['t_statistic']:.4f}")
            add_line(f"  p-value: {t['p_value']:.4f}")
            add_line(f"  Significant at α=0.05: {t['significant_0.05']}")
            add_line(f"  Significant at α=0.10: {t['significant_0.10']}")
        
        if 'cohens_d' in tests:
            add_line(f"\nEffect Size (Cohen's d): {tests['cohens_d']:.4f}")
            effect_interpretation = (
                "negligible" if abs(tests['cohens_d']) < 0.2 else
                "small" if abs(tests['cohens_d']) < 0.5 else
                "medium" if abs(tests['cohens_d']) < 0.8 else
                "large"
            )
            add_line(f"  Interpretation: {effect_interpretation}")
        
        add_line(f"\nAbsolute Improvement: {tests['absolute_improvement']:+.4f}")
        add_line(f"Relative Improvement: {tests['relative_improvement']:+.2f}%")
    
    add_line()
    
    # Per-Seed Comparison
    add_line("3. PER-SEED COMPARISON")
    add_line("-" * 40)
    add_line(f"{'Seed':<10} {'Baseline':<15} {'Bio-Inspired':<15} {'Improvement':<15} {'Winner':<10}")
    add_line("-" * 65)
    
    wins = {'baseline': 0, 'bio': 0}
    for c in comparisons:
        winner = "Bio" if c.get('bio_wins', False) else "Baseline"
        if c.get('bio_wins', False):
            wins['bio'] += 1
        else:
            wins['baseline'] += 1
        
        add_line(f"{c['seed']:<10} "
                f"{c['baseline_final_test_acc']:.4f}          "
                f"{c['bio_final_test_acc']:.4f}          "
                f"{c.get('improvement', 0):+.4f}          "
                f"{winner}")
    
    add_line("-" * 65)
    add_line(f"Win Count: Baseline={wins['baseline']}, Bio-Inspired={wins['bio']}")
    
    add_line()
    
    # Training Dynamics
    add_line("4. TRAINING DYNAMICS")
    add_line("-" * 40)
    
    if dynamics and 'baseline' in dynamics and 'refractory_lateral' in dynamics:
        base_dyn = dynamics['baseline']
        bio_dyn = dynamics['refractory_lateral']
        
        if 'convergence_epoch' in base_dyn and 'convergence_epoch' in bio_dyn:
            add_line(f"Convergence Epoch (to 95% of final):")
            add_line(f"  Baseline: {base_dyn['convergence_epoch']['mean']:.1f} ± {base_dyn['convergence_epoch']['std']:.1f}")
            add_line(f"  Bio-Inspired: {bio_dyn['convergence_epoch']['mean']:.1f} ± {bio_dyn['convergence_epoch']['std']:.1f}")
        
        if 'early_accuracy' in base_dyn and 'early_accuracy' in bio_dyn:
            add_line(f"\nEarly Accuracy (Epoch 20):")
            add_line(f"  Baseline: {base_dyn['early_accuracy']['mean']:.4f} ± {base_dyn['early_accuracy']['std']:.4f}")
            add_line(f"  Bio-Inspired: {bio_dyn['early_accuracy']['mean']:.4f} ± {bio_dyn['early_accuracy']['std']:.4f}")
        
        if 'mid_accuracy' in base_dyn and 'mid_accuracy' in bio_dyn:
            add_line(f"\nMid Accuracy (Epoch 50):")
            add_line(f"  Baseline: {base_dyn['mid_accuracy']['mean']:.4f} ± {base_dyn['mid_accuracy']['std']:.4f}")
            add_line(f"  Bio-Inspired: {bio_dyn['mid_accuracy']['mean']:.4f} ± {bio_dyn['mid_accuracy']['std']:.4f}")
        
        if 'accuracy_variance_over_training' in base_dyn and 'accuracy_variance_over_training' in bio_dyn:
            add_line(f"\nAccuracy Variance (2nd half of training):")
            add_line(f"  Baseline: {base_dyn['accuracy_variance_over_training']['mean']:.6f}")
            add_line(f"  Bio-Inspired: {bio_dyn['accuracy_variance_over_training']['mean']:.6f}")
            if bio_dyn['accuracy_variance_over_training']['mean'] < base_dyn['accuracy_variance_over_training']['mean']:
                add_line("  → Bio-inspired shows more stable convergence")
            else:
                add_line("  → Baseline shows more stable convergence")
    
    add_line()
    
    # Conclusion
    add_line("5. CONCLUSION")
    add_line("-" * 40)
    
    if tests and 'absolute_improvement' in tests:
        if tests['absolute_improvement'] > 0:
            add_line(f"✓ Refractory + Lateral Inhibition IMPROVES over baseline by {tests['absolute_improvement']:+.4f}")
        else:
            add_line(f"✗ Refractory + Lateral Inhibition DECREASES performance by {tests['absolute_improvement']:.4f}")
        
        if 'final_test_acc_ttest' in tests:
            if tests['final_test_acc_ttest']['significant_0.05']:
                add_line("✓ Result is STATISTICALLY SIGNIFICANT (p < 0.05)")
            elif tests['final_test_acc_ttest']['significant_0.10']:
                add_line("~ Result is marginally significant (p < 0.10)")
            else:
                add_line("✗ Result is NOT statistically significant (p >= 0.10)")
                add_line("  (May need more seeds to detect effect reliably)")
    
    add_line()
    add_line("=" * 80)
    
    # Save report to file
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Refractory + Lateral vs Baseline')
    parser.add_argument('--baseline_dir', type=str, default='outputs/bio_ablation',
                       help='Directory containing baseline experiments')
    parser.add_argument('--bio_dir', type=str, default='outputs/bio_experiments',
                       help='Directory containing bio-inspired experiments')
    parser.add_argument('--output_dir', type=str, default='outputs/analysis_refractory_lateral',
                       help='Directory to save analysis outputs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Seeds to analyze')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load experiments
    print(f"Loading experiments from:")
    print(f"  Baseline: {args.baseline_dir}")
    print(f"  Bio-inspired: {args.bio_dir}")
    print(f"  Seeds: {args.seeds}")
    print()
    
    experiments = load_all_experiments(args.baseline_dir, args.bio_dir, args.seeds)
    
    # Check what we loaded
    print(f"Loaded {len(experiments['baseline'])} baseline experiments")
    print(f"Loaded {len(experiments['refractory_lateral'])} bio-inspired experiments")
    print()
    
    if not experiments['baseline'] or not experiments['refractory_lateral']:
        print("ERROR: Could not load experiments. Check paths and file names.")
        return
    
    # Compute analyses
    summary = compute_summary_statistics(experiments)
    tests = compute_statistical_tests(summary)
    dynamics = analyze_training_dynamics(experiments)
    comparisons = analyze_per_seed_comparison(experiments)
    
    # Generate plots
    plot_training_curves(experiments, args.output_dir)
    plot_per_seed_comparison(comparisons, args.output_dir)
    plot_summary_statistics(summary, args.output_dir)
    plot_training_dynamics(dynamics, args.output_dir)
    
    # Print full report
    print_full_report(summary, tests, dynamics, comparisons, args.output_dir)
    
    # Save raw data
    analysis_data = {
        'summary': summary,
        'tests': tests,
        'dynamics': dynamics,
        'comparisons': comparisons
    }
    
    # Convert numpy arrays for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (bool, np.bool_)):  # Fix: handle booleans
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    with open(os.path.join(args.output_dir, 'analysis_data.json'), 'w') as f:
        json.dump(convert_to_serializable(analysis_data), f, indent=2)
    
    print(f"\nAll analysis outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()