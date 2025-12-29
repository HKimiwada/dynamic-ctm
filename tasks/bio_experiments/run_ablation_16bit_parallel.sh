#!/bin/bash
# tasks/bio_experiments/run_ablation_16bit_parallel.sh
# Comprehensive 16-bit parity ablation study for 8x V100 GPUs
set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR="outputs/bio_ablation_16bit"
PARITY_LENGTH=16
EPOCHS=100
EVAL_EVERY=1  # Evaluate every epoch for detailed learning curves

# More seeds for statistical power (your 32-bit had 3, let's use 5)
SEEDS=(42 123 456 789 1337)

# GPU configuration
GPUS=(0 1 2 3 4 5 6 7)
MAX_JOBS=${#GPUS[@]}

# ============================================================================
# CPU THROTTLING CONFIGURATION
# ============================================================================
TOTAL_CPUS=$(nproc)
CORES_PER_JOB=$(( TOTAL_CPUS / MAX_JOBS ))
if (( CORES_PER_JOB < 1 )); then CORES_PER_JOB=1; fi

# Set thread limits
export OMP_NUM_THREADS="$CORES_PER_JOB"
export MKL_NUM_THREADS="$CORES_PER_JOB"
export OPENBLAS_NUM_THREADS="$CORES_PER_JOB"
export VECLIB_MAXIMUM_THREADS="$CORES_PER_JOB"
export NUMEXPR_NUM_THREADS="$CORES_PER_JOB"
export TORCH_NUM_THREADS="$CORES_PER_JOB"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
core_range_for_slot() {
    local slot="$1"
    local start=$(( slot * CORES_PER_JOB ))
    local end=$(( start + CORES_PER_JOB - 1 ))
    local max_end=$(( TOTAL_CPUS - 1 ))
    if (( end > max_end )); then end="$max_end"; fi
    echo "${start}-${end}"
}

run_job() {
    local slot="$1"; shift
    local gpu="$1"; shift
    local cores
    cores="$(core_range_for_slot "$slot")"
    
    echo "[$(date '+%H:%M:%S')] [slot $slot | GPU $gpu | cores $cores] $*"
    
    CUDA_VISIBLE_DEVICES="$gpu" \
    taskset -c "$cores" \
    env \
        OMP_NUM_THREADS="$CORES_PER_JOB" \
        MKL_NUM_THREADS="$CORES_PER_JOB" \
        OPENBLAS_NUM_THREADS="$CORES_PER_JOB" \
        VECLIB_MAXIMUM_THREADS="$CORES_PER_JOB" \
        NUMEXPR_NUM_THREADS="$CORES_PER_JOB" \
        TORCH_NUM_THREADS="$CORES_PER_JOB" \
        "$@"
}

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 0.5
    done
}

job_idx=0
launch() {
    local slot=$(( job_idx % MAX_JOBS ))
    local gpu="${GPUS[$slot]}"
    job_idx=$((job_idx + 1))
    wait_for_slot
    ( run_job "$slot" "$gpu" "$@" ) &
}

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================
echo "============================================================================"
echo "16-bit Parity Bio-CTM Comprehensive Ablation Study"
echo "============================================================================"
echo "TOTAL_CPUS=$TOTAL_CPUS  MAX_JOBS=$MAX_JOBS  CORES_PER_JOB=$CORES_PER_JOB"
echo "GPU slots: ${GPUS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Output: $OUTPUT_DIR"
echo "============================================================================"

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# 1. BASELINE CTM (no bio mechanisms)
# ============================================================================
echo ""
echo ">>> Phase 1: Baseline CTM"
for SEED in "${SEEDS[@]}"; do
    launch python -m tasks.bio_experiments.train_16bit \
        --seed "$SEED" \
        --parity_length "$PARITY_LENGTH" \
        --epochs "$EPOCHS" \
        --eval_every "$EVAL_EVERY" \
        --output_dir "$OUTPUT_DIR" \
        --experiment_name "baseline_seed${SEED}"
done

# ============================================================================
# 2. FULL BIO-CTM (all mechanisms)
# ============================================================================
echo ""
echo ">>> Phase 2: Full Bio-CTM (all mechanisms)"
for SEED in "${SEEDS[@]}"; do
    launch python -m tasks.bio_experiments.train_16bit \
        --seed "$SEED" \
        --parity_length "$PARITY_LENGTH" \
        --epochs "$EPOCHS" \
        --eval_every "$EVAL_EVERY" \
        --output_dir "$OUTPUT_DIR" \
        --use_bio \
        --use_short_term_plasticity \
        --use_homeostasis \
        --use_lateral_inhibition \
        --use_refractory \
        --use_synaptic_noise \
        --experiment_name "full_bio_seed${SEED}"
done

# ============================================================================
# 3. SINGLE MECHANISM ABLATIONS (each mechanism alone)
# ============================================================================
echo ""
echo ">>> Phase 3: Single mechanism ablations"

MECHANISMS=(
    "use_short_term_plasticity:stp"
    "use_homeostasis:homeo"
    "use_lateral_inhibition:lateral"
    "use_refractory:refract"
    "use_synaptic_noise:noise"
)

for MECH_PAIR in "${MECHANISMS[@]}"; do
    MECH_FLAG="${MECH_PAIR%%:*}"
    MECH_NAME="${MECH_PAIR##*:}"
    
    for SEED in "${SEEDS[@]}"; do
        launch python -m tasks.bio_experiments.train_16bit \
            --seed "$SEED" \
            --parity_length "$PARITY_LENGTH" \
            --epochs "$EPOCHS" \
            --eval_every "$EVAL_EVERY" \
            --output_dir "$OUTPUT_DIR" \
            --use_bio \
            --"$MECH_FLAG" \
            --experiment_name "${MECH_NAME}_only_seed${SEED}"
    done
done

# ============================================================================
# 4. LEAVE-ONE-OUT ABLATIONS (full bio minus one)
# ============================================================================
echo ""
echo ">>> Phase 4: Leave-one-out ablations"

for MECH_PAIR in "${MECHANISMS[@]}"; do
    MECH_FLAG="${MECH_PAIR%%:*}"
    MECH_NAME="${MECH_PAIR##*:}"
    
    # Build args with all mechanisms EXCEPT this one
    ARGS=()
    for M_PAIR in "${MECHANISMS[@]}"; do
        M_FLAG="${M_PAIR%%:*}"
        if [ "$M_FLAG" != "$MECH_FLAG" ]; then
            ARGS+=( "--$M_FLAG" )
        fi
    done
    
    for SEED in "${SEEDS[@]}"; do
        launch python -m tasks.bio_experiments.train_16bit \
            --seed "$SEED" \
            --parity_length "$PARITY_LENGTH" \
            --epochs "$EPOCHS" \
            --eval_every "$EVAL_EVERY" \
            --output_dir "$OUTPUT_DIR" \
            --use_bio \
            "${ARGS[@]}" \
            --experiment_name "full_minus_${MECH_NAME}_seed${SEED}"
    done
done

# ============================================================================
# 5. KEY COMBINATIONS (based on your earlier finding that refractory shines)
# ============================================================================
echo ""
echo ">>> Phase 5: Key mechanism combinations"

# Refractory + Lateral (your earlier promising combo)
for SEED in "${SEEDS[@]}"; do
    launch python -m tasks.bio_experiments.train_16bit \
        --seed "$SEED" \
        --parity_length "$PARITY_LENGTH" \
        --epochs "$EPOCHS" \
        --eval_every "$EVAL_EVERY" \
        --output_dir "$OUTPUT_DIR" \
        --use_bio \
        --use_refractory \
        --use_lateral_inhibition \
        --experiment_name "refract_lateral_seed${SEED}"
done

# Refractory + Homeostasis
for SEED in "${SEEDS[@]}"; do
    launch python -m tasks.bio_experiments.train_16bit \
        --seed "$SEED" \
        --parity_length "$PARITY_LENGTH" \
        --epochs "$EPOCHS" \
        --eval_every "$EVAL_EVERY" \
        --output_dir "$OUTPUT_DIR" \
        --use_bio \
        --use_refractory \
        --use_homeostasis \
        --experiment_name "refract_homeo_seed${SEED}"
done

# Refractory + STP
for SEED in "${SEEDS[@]}"; do
    launch python -m tasks.bio_experiments.train_16bit \
        --seed "$SEED" \
        --parity_length "$PARITY_LENGTH" \
        --epochs "$EPOCHS" \
        --eval_every "$EVAL_EVERY" \
        --output_dir "$OUTPUT_DIR" \
        --use_bio \
        --use_refractory \
        --use_short_term_plasticity \
        --experiment_name "refract_stp_seed${SEED}"
done

# Lateral + Homeostasis (stability mechanisms)
for SEED in "${SEEDS[@]}"; do
    launch python -m tasks.bio_experiments.train_16bit \
        --seed "$SEED" \
        --parity_length "$PARITY_LENGTH" \
        --epochs "$EPOCHS" \
        --eval_every "$EVAL_EVERY" \
        --output_dir "$OUTPUT_DIR" \
        --use_bio \
        --use_lateral_inhibition \
        --use_homeostasis \
        --experiment_name "lateral_homeo_seed${SEED}"
done

# Triple combo: Refractory + Lateral + Homeostasis
for SEED in "${SEEDS[@]}"; do
    launch python -m tasks.bio_experiments.train_16bit \
        --seed "$SEED" \
        --parity_length "$PARITY_LENGTH" \
        --epochs "$EPOCHS" \
        --eval_every "$EVAL_EVERY" \
        --output_dir "$OUTPUT_DIR" \
        --use_bio \
        --use_refractory \
        --use_lateral_inhibition \
        --use_homeostasis \
        --experiment_name "refract_lateral_homeo_seed${SEED}"
done

# ============================================================================
# Wait for all jobs to complete
# ============================================================================
echo ""
echo ">>> Waiting for all jobs to complete..."
wait

echo ""
echo "============================================================================"
echo "Ablation study complete!"
echo "Total experiments: $job_idx"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================================"
echo ""
echo "Run analysis with:"
echo "  python -m tasks.bio_experiments.analyze_16bit --output_dir $OUTPUT_DIR"