import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
from scipy import stats

def load_detailed_results(output_dir):
    """Load full experiment results including time-series data."""
    results = {}
    for exp_name in os.listdir(output_dir):
        exp_path = os.path.join(output_dir, exp_name)
        if not os.path.isdir(exp_path): continue
            
        results_file = os.path.join(exp_path, 'results.json')
        config_file = os.path.join(exp_path, 'config.json')
        
        if os.path.exists(results_file) and os.path.exists(config_file):
            with open(results_file) as f: exp_results = json.load(f)
            with open(config_file) as f: config = json.load(f)
            results[exp_name] = {'results': exp_results, 'config': config}
    return results

def aggregate_data(results):
    """Aggregate all 39 runs by condition for statistical analysis[cite: 31]."""
    aggregated = defaultdict(lambda: {
        'test_acc_max': [], 'train_acc_max': [], 
        'test_series': [], 'epochs': None
    })
    
    for exp_name, data in results.items():
        parts = exp_name.rsplit('_seed', 1)
        condition = parts[0] if len(parts) > 1 else exp_name
        res = data['results']
        conf = data['config']
        
        if res['test_acc']:
            aggregated[condition]['test_acc_max'].append(max(res['test_acc']))
            aggregated[condition]['train_acc_max'].append(max(res['train_acc']))
            aggregated[condition]['test_series'].append(res['test_acc'])
            # Calculate x-axis based on eval_every [cite: 34]
            if aggregated[condition]['epochs'] is None:
                eval_every = conf.get('eval_every', 10)
                aggregated[condition]['epochs'] = np.arange(eval_every, conf['epochs'] + 1, eval_every)
    
    return aggregated

def plot_learning_curves(aggregated, output_dir):
    """1. Learning Curves: Mean + Shaded Variance for all seeds[cite: 34]."""
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    for condition, data in aggregated.items():
        if not data['test_series']: continue
        series = np.array(data['test_series'])
        mean = np.mean(series, axis=0)
        std = np.std(series, axis=0)
        epochs = data['epochs'][:len(mean)]
        
        line = plt.plot(epochs, mean, label=condition, linewidth=2)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=line[0].get_color())

    plt.title('Learning Dynamics (Mean Test Accuracy ± Std Dev)')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=200)
    plt.close()

def plot_ablation_summary(aggregated, output_dir):
    """2 & 4. Summary Bar Chart with Seed Points and Significance Markers[cite: 31]."""
    conditions = sorted(aggregated.keys())
    baseline_accs = aggregated.get('baseline', {}).get('test_acc_max', [])
    
    means = [np.mean(aggregated[c]['test_acc_max']) for c in conditions]
    stds = [np.std(aggregated[c]['test_acc_max']) for c in conditions]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(conditions))
    
    # Base bars
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.4, color='skyblue', label='Mean Performance')
    
    # 2. Seed Swarm Plot: Overlay individual seed points
    for i, cond in enumerate(conditions):
        seed_data = aggregated[cond]['test_acc_max']
        ax.scatter([i] * len(seed_data), seed_data, color='black', alpha=0.6, zorder=3, s=20)
        
        # 4. Statistical Significance (T-test vs Baseline)
        if cond != 'baseline' and len(baseline_accs) > 1 and len(seed_data) > 1:
            t_stat, p_val = stats.ttest_ind(seed_data, baseline_accs)
            if p_val < 0.05:
                marker = '**' if p_val < 0.01 else '*'
                ax.text(i, max(seed_data) + 0.02, marker, ha='center', fontsize=15, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Peak Test Accuracy')
    ax.set_title('Ablation Summary: Impact of Bio-mechanisms (* p < 0.05 vs Baseline)')
    
    if baseline_accs:
        ax.axhline(np.mean(baseline_accs), color='gray', linestyle='--', alpha=0.8, label='Baseline Mean')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_with_stats.png'), dpi=200)
    plt.close()

def plot_overfitting_analysis(aggregated, output_dir):
    """3. Overfitting Analysis: Comparing Train vs Test gaps."""
    conditions = sorted(aggregated.keys())
    train_means = [np.mean(aggregated[c]['train_acc_max']) for c in conditions]
    test_means = [np.mean(aggregated[c]['test_acc_max']) for c in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, train_means, width, label='Final Train Acc', color='#2ecc71', alpha=0.7)
    ax.bar(x + width/2, test_means, width, label='Final Test Acc', color='#3498db', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Generalization Analysis: Train vs. Test Accuracy')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overfitting_analysis.png'), dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs/bio_ablation')
    args = parser.parse_args()
    
    raw_results = load_detailed_results(args.output_dir)
    if not raw_results:
        print("No results found.")
        return
        
    aggregated = aggregate_data(raw_results)
    
    # Run the four requested detailed analyses
    print("Generating Learning Curves...")
    plot_learning_curves(aggregated, args.output_dir)
    
    print("Generating Summary with Seed Points and Significance...")
    plot_ablation_summary(aggregated, args.output_dir)
    
    print("Generating Overfitting Analysis...")
    plot_overfitting_analysis(aggregated, args.output_dir)
    
    print(f"Detailed analysis complete. Files saved to {args.output_dir}")

if __name__ == '__main__':
    main()