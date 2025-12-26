#!/bin/bash
# tasks/bio_experiments/run_ablation_parallel.sh
set -euo pipefail

OUTPUT_DIR="outputs/bio_ablation"
PARITY_LENGTH=32
EPOCHS=100
SEEDS=(42 123 456)

# --- GPU + concurrency ---
GPUS=(0 1 2 3 4 5 6 7)
MAX_JOBS=${#GPUS[@]}

# --- CPU throttling (nproc=80) ---
# Allocate a fixed CPU slice per job and pin each job to its own cores.
# With 8 jobs: 80 / 8 = 10 cores per job.
TOTAL_CPUS=$(nproc)
CORES_PER_JOB=$(( TOTAL_CPUS / MAX_JOBS ))   # 10
if (( CORES_PER_JOB < 1 )); then CORES_PER_JOB=1; fi

# Also cap common math / dataloader thread pools per process.
# Set these to <= CORES_PER_JOB (usually equal is fine).
export OMP_NUM_THREADS="$CORES_PER_JOB"
export MKL_NUM_THREADS="$CORES_PER_JOB"
export OPENBLAS_NUM_THREADS="$CORES_PER_JOB"
export VECLIB_MAXIMUM_THREADS="$CORES_PER_JOB"
export NUMEXPR_NUM_THREADS="$CORES_PER_JOB"

# Optional: limit PyTorch intra/inter-op threads if you use torch (harmless otherwise).
export TORCH_NUM_THREADS="$CORES_PER_JOB"

# Build core ranges for each GPU slot: slot i gets cores [i*CORES_PER_JOB, (i+1)*CORES_PER_JOB-1]
core_range_for_slot () {
  local slot="$1"
  local start=$(( slot * CORES_PER_JOB ))
  local end=$(( start + CORES_PER_JOB - 1 ))

  # clamp end to TOTAL_CPUS-1 (handles non-multiple edge cases)
  local max_end=$(( TOTAL_CPUS - 1 ))
  if (( end > max_end )); then end="$max_end"; fi

  echo "${start}-${end}"
}

run_job () {
  local slot="$1"; shift
  local gpu="$1"; shift
  local cores
  cores="$(core_range_for_slot "$slot")"

  echo "[slot $slot | GPU $gpu | cores $cores] $*"

  # Pin process to dedicated CPU cores + bind memory locally
  # (numactl is nice if you have it, but taskset works everywhere)
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

# simple semaphore: wait if >= MAX_JOBS background jobs
wait_for_slot () {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 0.2
  done
}

job_idx=0
launch () {
  local slot=$(( job_idx % MAX_JOBS ))
  local gpu="${GPUS[$slot]}"
  job_idx=$((job_idx + 1))
  wait_for_slot
  ( run_job "$slot" "$gpu" "$@" ) &
}

echo "TOTAL_CPUS=$TOTAL_CPUS  MAX_JOBS=$MAX_JOBS  CORES_PER_JOB=$CORES_PER_JOB"
echo "GPU slots: ${GPUS[*]}"

# 1. Baseline CTM
for SEED in "${SEEDS[@]}"; do
  launch python -m tasks.bio_experiments.train \
    --seed "$SEED" \
    --parity_length "$PARITY_LENGTH" \
    --epochs "$EPOCHS" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "baseline_seed${SEED}"
done

# 2. Full Bio-CTM
for SEED in "${SEEDS[@]}"; do
  launch python -m tasks.bio_experiments.train \
    --seed "$SEED" \
    --parity_length "$PARITY_LENGTH" \
    --epochs "$EPOCHS" \
    --output_dir "$OUTPUT_DIR" \
    --use_bio \
    --use_short_term_plasticity \
    --use_homeostasis \
    --use_lateral_inhibition \
    --use_refractory \
    --use_synaptic_noise \
    --experiment_name "full_bio_seed${SEED}"
done

# 3. Individual mechanism ablations
MECHANISMS=(use_short_term_plasticity use_homeostasis use_lateral_inhibition use_refractory use_synaptic_noise)

for MECH in "${MECHANISMS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    launch python -m tasks.bio_experiments.train \
      --seed "$SEED" \
      --parity_length "$PARITY_LENGTH" \
      --epochs "$EPOCHS" \
      --output_dir "$OUTPUT_DIR" \
      --use_bio \
      --"$MECH" \
      --experiment_name "${MECH}_only_seed${SEED}"
  done
done

# 4. Full bio minus one mechanism
for MECH in "${MECHANISMS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    ARGS=()
    for M in "${MECHANISMS[@]}"; do
      if [ "$M" != "$MECH" ]; then
        ARGS+=( "--$M" )
      fi
    done

    launch python -m tasks.bio_experiments.train \
      --seed "$SEED" \
      --parity_length "$PARITY_LENGTH" \
      --epochs "$EPOCHS" \
      --output_dir "$OUTPUT_DIR" \
      --use_bio \
      "${ARGS[@]}" \
      --experiment_name "full_minus_${MECH}_seed${SEED}"
  done
done

wait
echo "Ablation study complete!"
