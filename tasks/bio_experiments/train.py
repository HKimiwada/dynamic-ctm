# tasks/bio_experiments/train.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

from models.ctm_bio import BioInspiredCTM
from models.ctm import ContinuousThoughtMachine
from data.custom_datasets import ParityDataset
from utils.losses import parity_loss
from utils.housekeeping import set_seed


def get_args():
    parser = argparse.ArgumentParser()
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='outputs/bio_experiments')
    parser.add_argument('--experiment_name', type=str, default=None)
    
    # Task settings
    parser.add_argument('--task', type=str, default='parity', choices=['parity', 'cifar10', 'maze'])
    parser.add_argument('--parity_length', type=int, default=32)
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_every', type=int, default=10)
    
    # CTM architecture
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_input', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--n_synch_out', type=int, default=32)
    parser.add_argument('--n_synch_action', type=int, default=32)
    parser.add_argument('--synapse_depth', type=int, default=2)
    parser.add_argument('--memory_length', type=int, default=10)
    parser.add_argument('--memory_hidden_dims', type=int, default=32)
    
    # Bio-inspired flags (for ablation)
    parser.add_argument('--use_bio', action='store_true', help='Use bio-inspired CTM')
    parser.add_argument('--use_short_term_plasticity', action='store_true')
    parser.add_argument('--use_homeostasis', action='store_true')
    parser.add_argument('--use_lateral_inhibition', action='store_true')
    parser.add_argument('--use_refractory', action='store_true')
    parser.add_argument('--use_synaptic_noise', action='store_true')
    
    # Bio-inspired hyperparameters
    parser.add_argument('--tau_facilitate', type=float, default=5.0)
    parser.add_argument('--tau_depress', type=float, default=20.0)
    parser.add_argument('--target_firing_rate', type=float, default=0.5)
    parser.add_argument('--homeostasis_adaptation_rate', type=float, default=0.01)
    parser.add_argument('--inhibition_strength', type=float, default=0.1)
    parser.add_argument('--inhibition_neighborhood', type=int, default=8)
    parser.add_argument('--refractory_strength', type=float, default=0.3)
    parser.add_argument('--refractory_decay', type=float, default=0.8)
    parser.add_argument('--noise_scale', type=float, default=0.01)
    
    return parser.parse_args()


def build_model(args):
    """Build either baseline CTM or Bio-Inspired CTM based on args."""
    
    ctm_kwargs = dict(
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=True,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=False,
        backbone_type='parity_backbone',
        positional_embedding_type='custom-rotational-1d',
        out_dims=args.parity_length * 2,
        prediction_reshaper=[args.parity_length, 2],
        dropout=0.0,
        neuron_select_type='random-pairing',
        n_random_pairing_self=0,
    )
    
    if args.use_bio:
        model = BioInspiredCTM(
            use_short_term_plasticity=args.use_short_term_plasticity,
            use_homeostasis=args.use_homeostasis,
            use_lateral_inhibition=args.use_lateral_inhibition,
            use_refractory=args.use_refractory,
            use_synaptic_noise=args.use_synaptic_noise,
            tau_facilitate=args.tau_facilitate,
            tau_depress=args.tau_depress,
            target_firing_rate=args.target_firing_rate,
            homeostasis_adaptation_rate=args.homeostasis_adaptation_rate,
            inhibition_strength=args.inhibition_strength,
            inhibition_neighborhood=args.inhibition_neighborhood,
            refractory_strength=args.refractory_strength,
            refractory_decay=args.refractory_decay,
            noise_scale=args.noise_scale,
            **ctm_kwargs
        )
    else:
        model = ContinuousThoughtMachine(**ctm_kwargs)
    
    return model


def compute_accuracy(predictions, targets, prediction_reshaper):
    """Compute accuracy at most certain time step."""
    B = predictions.size(0)
    preds_reshaped = predictions.reshape([B] + prediction_reshaper + [predictions.size(-1)])
    # Use final time step for simplicity
    final_preds = preds_reshaped[..., -1].argmax(dim=-1)  # (B, parity_length)
    correct = (final_preds == targets).float().mean()
    return correct.item()


def train_epoch(model, dataloader, optimizer, device, prediction_reshaper):
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions, certainties, _ = model(inputs)
        
        loss, _ = parity_loss(
            predictions.reshape([inputs.size(0)] + prediction_reshaper + [predictions.size(-1)]),
            certainties,
            targets,
            use_most_certain=True
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += compute_accuracy(predictions, targets, prediction_reshaper)
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


@torch.no_grad()
def evaluate(model, dataloader, device, prediction_reshaper):
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        predictions, certainties, _ = model(inputs)
        
        loss, _ = parity_loss(
            predictions.reshape([inputs.size(0)] + prediction_reshaper + [predictions.size(-1)]),
            certainties,
            targets,
            use_most_certain=True
        )
        
        total_loss += loss.item()
        total_acc += compute_accuracy(predictions, targets, prediction_reshaper)
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def main():
    args = get_args()
    set_seed(args.seed)
    
    # Setup output directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bio_str = 'bio' if args.use_bio else 'baseline'
        args.experiment_name = f'{args.task}_{bio_str}_{timestamp}'
    
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Build dataset
    prediction_reshaper = [args.parity_length, 2]
    train_dataset = ParityDataset(sequence_length=args.parity_length, length=10000)
    test_dataset = ParityDataset(sequence_length=args.parity_length, length=1000)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Build model
    model = build_model(args).to(args.device)
    
    # Dummy forward to initialize lazy modules
    dummy_input = torch.randint(0, 2, (1, args.parity_length), device=args.device).float() * 2 - 1
    _ = model(dummy_input)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    
    best_test_acc = 0
    
    for epoch in tqdm(range(args.epochs), desc='Training'):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.device, prediction_reshaper
        )
        scheduler.step()
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        
        if (epoch + 1) % args.eval_every == 0:
            test_loss, test_acc = evaluate(model, test_loader, args.device, prediction_reshaper)
            results['test_loss'].append(test_loss)
            results['test_acc'].append(test_acc)
            
            print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'args': vars(args),
                }, os.path.join(output_dir, 'best_model.pt'))
    
    # Save final results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBest test accuracy: {best_test_acc:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()