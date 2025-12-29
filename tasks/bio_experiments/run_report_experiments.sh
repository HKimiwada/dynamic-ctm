#!/bin/bash
# Bio-CTM Report Experiments
# Run all experiments needed for a comprehensive research report

set -e

# Configuration
SEEDS=(1 2 3 4 5 6 7 8 9 10)
OUTPUT_BASE="outputs/bio_report"
N_EPOCHS=100

echo "========================================"
echo "Bio-CTM Research Report Experiments"
echo "========================================"

mkdir -p $OUTPUT_BASE

# ============================================
# EXPERIMENT 1: 32-bit Parity (10 seeds)
# ============================================
echo ""
echo "=== EXPERIMENT 1: 32-bit Parity (10 seeds) ==="
echo ""

NUM_GPUS=8
JOB_ID=0

for SEED in "${SEEDS[@]}"; do
    GPU_ID=$((JOB_ID % NUM_GPUS))
    echo "Running 32-bit baseline seed $SEED on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m tasks.bio_experiments.train \
        --parity_length 32 \
        --seed $SEED \
        --epochs $N_EPOCHS \
        --experiment_name "32bit_baseline_seed${SEED}" \
        --output_dir "${OUTPUT_BASE}/32bit_parity" &
    JOB_ID=$((JOB_ID + 1))
    
    GPU_ID=$((JOB_ID % NUM_GPUS))
    echo "Running 32-bit full_minus_homeo seed $SEED on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m tasks.bio_experiments.train \
        --parity_length 32 \
        --seed $SEED \
        --epochs $N_EPOCHS \
        --use_bio \
        --use_refractory \
        --use_lateral_inhibition \
        --use_short_term_plasticity \
        --use_synaptic_noise \
        --experiment_name "32bit_full_minus_homeo_seed${SEED}" \
        --output_dir "${OUTPUT_BASE}/32bit_parity" &
    JOB_ID=$((JOB_ID + 1))
    
    GPU_ID=$((JOB_ID % NUM_GPUS))
    echo "Running 32-bit refract_only seed $SEED on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m tasks.bio_experiments.train \
        --parity_length 32 \
        --seed $SEED \
        --epochs $N_EPOCHS \
        --use_bio \
        --use_refractory \
        --experiment_name "32bit_refract_only_seed${SEED}" \
        --output_dir "${OUTPUT_BASE}/32bit_parity" &
    JOB_ID=$((JOB_ID + 1))
    
    # Wait after every 8 jobs (one full round of GPUs)
    if (( JOB_ID % NUM_GPUS == 0 )); then
        wait
    fi
done
wait

echo "32-bit parity experiments complete!"

# ============================================
# EXPERIMENT 2: 16-bit Parity Validation (10 seeds for top configs)
# ============================================
echo ""
echo "=== EXPERIMENT 2: 16-bit Parity Extended Validation ==="
echo ""

JOB_ID=0

for SEED in "${SEEDS[@]}"; do
    GPU_ID=$((JOB_ID % NUM_GPUS))
    echo "Running 16-bit baseline seed $SEED on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m tasks.bio_experiments.train \
        --parity_length 16 \
        --seed $SEED \
        --epochs $N_EPOCHS \
        --experiment_name "16bit_baseline_seed${SEED}" \
        --output_dir "${OUTPUT_BASE}/16bit_validation" &
    JOB_ID=$((JOB_ID + 1))
    
    GPU_ID=$((JOB_ID % NUM_GPUS))
    echo "Running 16-bit full_minus_homeo seed $SEED on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m tasks.bio_experiments.train \
        --parity_length 16 \
        --seed $SEED \
        --epochs $N_EPOCHS \
        --use_bio \
        --use_refractory \
        --use_lateral_inhibition \
        --use_short_term_plasticity \
        --use_synaptic_noise \
        --experiment_name "16bit_full_minus_homeo_seed${SEED}" \
        --output_dir "${OUTPUT_BASE}/16bit_validation" &
    JOB_ID=$((JOB_ID + 1))
    
    # Wait after every 8 jobs
    if (( JOB_ID % NUM_GPUS == 0 )); then
        wait
    fi
done
wait

echo "16-bit validation experiments complete!"

# ============================================
# EXPERIMENT 3: Hyperparameter Sensitivity
# ============================================
echo ""
echo "=== EXPERIMENT 3: Hyperparameter Sensitivity ==="
echo ""

REFRACTORY_STRENGTHS=(0.1 0.2 0.3 0.4 0.5)
INHIBITION_STRENGTHS=(0.05 0.1 0.15 0.2)

JOB_ID=0

for REF in "${REFRACTORY_STRENGTHS[@]}"; do
    for LAT in "${INHIBITION_STRENGTHS[@]}"; do
        GPU_ID=$((JOB_ID % NUM_GPUS))
        echo "Running hparam sweep: ref=$REF, lat=$LAT on GPU $GPU_ID..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m tasks.bio_experiments.train \
            --parity_length 16 \
            --seed 42 \
            --epochs $N_EPOCHS \
            --use_bio \
            --use_refractory --refractory_strength $REF \
            --use_lateral_inhibition --inhibition_strength $LAT \
            --use_short_term_plasticity \
            --use_synaptic_noise \
            --experiment_name "hparam_ref${REF}_lat${LAT}" \
            --output_dir "${OUTPUT_BASE}/hparam_sweep" &
        JOB_ID=$((JOB_ID + 1))
        
        # Wait after every 8 jobs
        if (( JOB_ID % NUM_GPUS == 0 )); then
            wait
        fi
    done
done
wait

echo "Hyperparameter sweep complete!"

# ============================================
# EXPERIMENT 4: Computational Cost Analysis
# ============================================
echo ""
echo "=== EXPERIMENT 4: Computational Cost Analysis ==="
echo ""

# Run timing experiments
python -c "
import time
import torch
import json
import os

# Setup
os.makedirs('${OUTPUT_BASE}/timing', exist_ok=True)
results = {}

# Import your modules
try:
    from tasks.bio_experiments.train import create_model, create_dataloader
    
    # Time baseline
    model = create_model(use_bio=False)
    model.cuda()
    
    # Warmup
    x = torch.randn(32, 16).cuda()
    for _ in range(10):
        _ = model(x)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model(x)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / 100
    
    results['baseline_inference_ms'] = baseline_time * 1000
    
    # Time bio model
    model = create_model(use_bio=True, use_refractory=True, 
                         use_lateral_inhibition=True, use_stp=True, 
                         use_synaptic_noise=True)
    model.cuda()
    
    for _ in range(10):
        _ = model(x)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model(x)
    torch.cuda.synchronize()
    bio_time = (time.time() - start) / 100
    
    results['bio_inference_ms'] = bio_time * 1000
    results['overhead_pct'] = (bio_time - baseline_time) / baseline_time * 100
    
    with open('${OUTPUT_BASE}/timing/timing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Baseline: {results[\"baseline_inference_ms\"]:.3f} ms')
    print(f'Bio-CTM: {results[\"bio_inference_ms\"]:.3f} ms')
    print(f'Overhead: {results[\"overhead_pct\"]:.1f}%')
    
except Exception as e:
    print(f'Timing analysis skipped: {e}')
    results['error'] = str(e)
    with open('${OUTPUT_BASE}/timing/timing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
"

echo "Timing analysis complete!"

# ============================================
# SUMMARY
# ============================================
echo ""
echo "========================================"
echo "All experiments complete!"
echo "========================================"
echo ""
echo "Results saved to: ${OUTPUT_BASE}/"
echo ""
echo "Next steps:"
echo "  1. Run analysis: python bio_ctm_report_analysis.py --results_dir ${OUTPUT_BASE}/16bit_validation"
echo "  2. Generate 32-bit analysis: python bio_ctm_report_analysis.py --results_dir ${OUTPUT_BASE}/32bit_parity --task_name '32-bit Parity'"
echo "  3. Analyze hparam sweep: python analyze_hparam_sweep.py --results_dir ${OUTPUT_BASE}/hparam_sweep"
echo ""