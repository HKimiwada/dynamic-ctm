#!/bin/bash
# tasks/bio_experiments/quick_test_16bit.sh
# Quick sanity check before running full ablation (2 seeds, fewer epochs)
set -euo pipefail

OUTPUT_DIR="outputs/bio_ablation_16bit_test"
PARITY_LENGTH=16
EPOCHS=20
SEEDS=(42 123)

echo "Quick test: 16-bit parity Bio-CTM"
echo "================================="
echo "This runs a minimal test to verify everything works before the full ablation."
echo ""

mkdir -p "$OUTPUT_DIR"

# Test 1: Baseline
echo ">>> Testing baseline CTM..."
python -m tasks.bio_experiments.train_16bit \
    --seed 42 \
    --parity_length "$PARITY_LENGTH" \
    --epochs "$EPOCHS" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "test_baseline_seed42"

# Test 2: Single mechanism (refractory - your best performer)
echo ""
echo ">>> Testing refractory-only Bio-CTM..."
python -m tasks.bio_experiments.train_16bit \
    --seed 42 \
    --parity_length "$PARITY_LENGTH" \
    --epochs "$EPOCHS" \
    --output_dir "$OUTPUT_DIR" \
    --use_bio \
    --use_refractory \
    --experiment_name "test_refract_seed42"

# Test 3: Full bio
echo ""
echo ">>> Testing full Bio-CTM..."
python -m tasks.bio_experiments.train_16bit \
    --seed 42 \
    --parity_length "$PARITY_LENGTH" \
    --epochs "$EPOCHS" \
    --output_dir "$OUTPUT_DIR" \
    --use_bio \
    --use_short_term_plasticity \
    --use_homeostasis \
    --use_lateral_inhibition \
    --use_refractory \
    --use_synaptic_noise \
    --experiment_name "test_full_bio_seed42"

echo ""
echo "================================="
echo "Quick test complete!"
echo "Check $OUTPUT_DIR for results"
echo ""
echo "If everything looks good, run the full ablation:"
echo "  bash tasks/bio_experiments/run_ablation_16bit_parallel.sh"