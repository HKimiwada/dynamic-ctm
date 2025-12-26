# tasks/bio_experiments/analyze_results.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse


def load_results(output_dir):
    """Load all experiment results from output directory."""
    results = {}
    
    for exp_name in os.listdir(output_dir):
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


def aggregate_by_condition(results):
    """Aggregate results by experimental condition (averaging over seeds)."""
    aggregated = defaultdict(lambda: {'test_acc': [], 'train_acc': []})
    
    for exp_name, data in results.items():
        # Extract condition name (remove seed suffix)
        parts = exp_name.rsplit('_seed', 1)
        condition = parts[0] if len(parts) > 1 else exp_name
        
        if data['results']['test_acc']:
            aggregated[condition]['test_acc'].append(max(data['results']['test_acc']))
        if data['results']['train_acc']:
            aggregated[condition]['train_acc'].append(max(data['results']['train_acc']))
    
    # Compute mean and std
    summary = {}
    for condition, accs in aggregated.items():
        summary[condition] = {
            'test_acc_mean': np.mean(accs['test_acc']) if accs['test_acc'] else 0,
            'test_acc_std': np.std(accs['test_acc']) if accs['test_acc'] else 0,
            'train_acc_mean': np.mean(accs['train_acc']) if accs['train_acc'] else 0,
            'train_acc_std': np.std(accs['train_acc']) if accs['train_acc'] else 0,
            'n_runs': len(accs['test_acc'])
        }
    
    return summary


def plot_ablation_results(summary, output_path):
    """Create bar plot of ablation results."""
    conditions = sorted(summary.keys())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(conditions))
    means = [summary[c]['test_acc_mean'] for c in conditions]
    stds = [summary[c]['test_acc_std'] for c in conditions]
    
    bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8)
    
    # Color baseline differently
    for i, cond in enumerate(conditions):
        if 'baseline' in cond.lower():
            bars[i].set_color('gray')
        elif 'full_bio' in cond.lower():
            bars[i].set_color('green')
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Bio-Inspired CTM Ablation Study')
    ax.set_ylim(0, 1.05)
    
    # Add horizontal line at baseline
    baseline_acc = summary.get('baseline', {}).get('test_acc_mean', 0)
    if baseline_acc > 0:
        ax.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.7, label='Baseline')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ablation plot to {output_path}")


def print_summary_table(summary):
    """Print formatted summary table."""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Condition':<40} {'Test Acc':>12} {'Train Acc':>12} {'N':>5}")
    print("-"*80)
    
    for condition in sorted(summary.keys()):
        s = summary[condition]
        test_str = f"{s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f}"
        train_str = f"{s['train_acc_mean']:.4f} ± {s['train_acc_std']:.4f}"
        print(f"{condition:<40} {test_str:>12} {train_str:>12} {s['n_runs']:>5}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs/bio_ablation')
    args = parser.parse_args()
    
    results = load_results(args.output_dir)
    
    if not results:
        print(f"No results found in {args.output_dir}")
        return
    
    summary = aggregate_by_condition(results)
    print_summary_table(summary)
    
    plot_path = os.path.join(args.output_dir, 'ablation_results.png')
    plot_ablation_results(summary, plot_path)


if __name__ == '__main__':
    main()