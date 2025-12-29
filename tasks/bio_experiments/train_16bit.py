# tasks/bio_experiments/train_16bit.py
"""
Training script for 16-bit parity task - where bio-inspired mechanisms shine.
Optimized for systematic ablation studies.
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.ctm_bio import BioInspiredCTM
from models.ctm import ContinuousThoughtMachine
from data.custom_datasets import ParityDataset
from utils.losses import parity_loss
from utils.housekeeping import set_seed


def get_args():
    parser = argparse.ArgumentParser(description='16-bit Parity Bio-CTM Ablation')
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='outputs/bio_ablation_16bit')
    parser.add_argument('--experiment_name', type=str, default=None)
    
    # Task settings - 16-bit is the sweet spot
    parser.add_argument('--parity_length', type=int, default=16)
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=2000)
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--eval_every', type=int, default=1)  # Evaluate every epoch for detailed curves
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # CTM architecture (tuned for 16-bit parity)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_input', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--n_synch_out', type=int, default=32)
    parser.add_argument('--n_synch_action', type=int, default=32)
    parser.add_argument('--synapse_depth', type=int, default=2)
    parser.add_argument('--memory_length', type=int, default=10)
    parser.add_argument('--memory_hidden_dims', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Bio-inspired flags (for ablation)
    parser.add_argument('--use_bio', action='store_true', help='Use bio-inspired CTM')
    parser.add_argument('--use_short_term_plasticity', action='store_true')
    parser.add_argument('--use_homeostasis', action='store_true')
    parser.add_argument('--use_lateral_inhibition', action='store_true')
    parser.add_argument('--use_refractory', action='store_true')
    parser.add_argument('--use_synaptic_noise', action='store_true')
    
    # Bio-inspired hyperparameters (can tune these)
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
        dropout=args.dropout,
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
    """Compute accuracy at final time step."""
    B = predictions.size(0)
    preds_reshaped = predictions.reshape([B] + prediction_reshaper + [predictions.size(-1)])
    final_preds = preds_reshaped[..., -1].argmax(dim=-1)
    correct = (final_preds == targets).float().mean()
    return correct.item()


def compute_most_certain_accuracy(predictions, certainties, targets, prediction_reshaper):
    """Compute accuracy at the most certain time step."""
    B = predictions.size(0)
    preds_reshaped = predictions.reshape([B] + prediction_reshaper + [predictions.size(-1)])
    
    # Get most certain time step per sample
    # certainties shape: (B, 2, T) where [:, 1, :] is confidence
    most_certain_t = certainties[:, 1, :].argmax(dim=-1)  # (B,)
    
    # Gather predictions at most certain time step
    batch_idx = torch.arange(B, device=predictions.device)
    most_certain_preds = preds_reshaped[batch_idx, :, :, most_certain_t]  # (B, parity_length, 2)
    final_preds = most_certain_preds.argmax(dim=-1)  # (B, parity_length)
    
    correct = (final_preds == targets).float().mean()
    return correct.item()


def train_epoch(model, dataloader, optimizer, device, prediction_reshaper, grad_clip=1.0):
    model.train()
    total_loss = 0
    total_acc = 0
    total_most_certain_acc = 0
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
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += compute_accuracy(predictions, targets, prediction_reshaper)
        total_most_certain_acc += compute_most_certain_accuracy(predictions, certainties, targets, prediction_reshaper)
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'acc': total_acc / num_batches,
        'most_certain_acc': total_most_certain_acc / num_batches
    }


@torch.no_grad()
def evaluate(model, dataloader, device, prediction_reshaper):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_most_certain_acc = 0
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
        total_most_certain_acc += compute_most_certain_accuracy(predictions, certainties, targets, prediction_reshaper)
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'acc': total_acc / num_batches,
        'most_certain_acc': total_most_certain_acc / num_batches
    }


def get_active_mechanisms(args):
    """Return list of active bio mechanisms for logging."""
    mechanisms = []
    if args.use_short_term_plasticity:
        mechanisms.append('stp')
    if args.use_homeostasis:
        mechanisms.append('homeo')
    if args.use_lateral_inhibition:
        mechanisms.append('lateral')
    if args.use_refractory:
        mechanisms.append('refract')
    if args.use_synaptic_noise:
        mechanisms.append('noise')
    return mechanisms


def main():
    args = get_args()
    set_seed(args.seed)
    
    # Setup output directory
    if args.experiment_name is None:
        mechanisms = get_active_mechanisms(args)
        if args.use_bio and mechanisms:
            mech_str = '_'.join(mechanisms)
        elif args.use_bio:
            mech_str = 'bio_none'
        else:
            mech_str = 'baseline'
        args.experiment_name = f'{mech_str}_seed{args.seed}'
    
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config = vars(args).copy()
    config['active_mechanisms'] = get_active_mechanisms(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Active mechanisms: {get_active_mechanisms(args) or ['baseline']}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)
    
    # Build dataset
    prediction_reshaper = [args.parity_length, 2]
    train_dataset = ParityDataset(sequence_length=args.parity_length, length=args.train_size)
    test_dataset = ParityDataset(sequence_length=args.parity_length, length=args.test_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    
    # Build model
    model = build_model(args).to(args.device)
    
    # Dummy forward to initialize lazy modules
    dummy_input = torch.randint(0, 2, (1, args.parity_length), device=args.device).float() * 2 - 1
    _ = model(dummy_input)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    results = {
        'train_loss': [],
        'train_acc': [],
        'train_most_certain_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_most_certain_acc': [],
        'epoch_times': [],
        'learning_rates': [],
    }
    
    best_test_acc = 0
    start_time = time.time()
    
    pbar = tqdm(range(args.epochs), desc='Training')
    for epoch in pbar:
        epoch_start = time.time()
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, prediction_reshaper, args.grad_clip
        )
        scheduler.step()
        
        results['train_loss'].append(train_metrics['loss'])
        results['train_acc'].append(train_metrics['acc'])
        results['train_most_certain_acc'].append(train_metrics['most_certain_acc'])
        results['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            test_metrics = evaluate(model, test_loader, args.device, prediction_reshaper)
            results['test_loss'].append(test_metrics['loss'])
            results['test_acc'].append(test_metrics['acc'])
            results['test_most_certain_acc'].append(test_metrics['most_certain_acc'])
            
            # Update progress bar
            pbar.set_postfix({
                'train_acc': f"{train_metrics['acc']:.3f}",
                'test_acc': f"{test_metrics['acc']:.3f}",
                'loss': f"{train_metrics['loss']:.4f}"
            })
            
            # Save best model
            if test_metrics['acc'] > best_test_acc:
                best_test_acc = test_metrics['acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_metrics['acc'],
                    'args': vars(args),
                }, os.path.join(output_dir, 'best_model.pt'))
        
        epoch_time = time.time() - epoch_start
        results['epoch_times'].append(epoch_time)
    
    total_time = time.time() - start_time
    
    # Final evaluation
    final_test = evaluate(model, test_loader, args.device, prediction_reshaper)
    
    # Save final results
    results['best_test_acc'] = best_test_acc
    results['final_test_acc'] = final_test['acc']
    results['final_test_most_certain_acc'] = final_test['most_certain_acc']
    results['total_training_time'] = total_time
    results['num_parameters'] = num_params
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': final_test['acc'],
        'args': vars(args),
    }, os.path.join(output_dir, 'final_model.pt'))
    
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Final test accuracy: {final_test['acc']:.4f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to {output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()