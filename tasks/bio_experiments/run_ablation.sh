#!/bin/bash
# tasks/bio_experiments/run_ablation.sh

OUTPUT_DIR="outputs/bio_ablation"
PARITY_LENGTH=32
EPOCHS=100
SEEDS="42 123 456"

# 1. Baseline CTM
for SEED in $SEEDS; do
    python -m tasks.bio_experiments.train \
        --seed $SEED \
        --parity_length $PARITY_LENGTH \
        --epochs $EPOCHS \
        --output_dir $OUTPUT_DIR \
        --experiment_name "baseline_seed${SEED}"
done

# 2. Full Bio-CTM (all mechanisms)
for SEED in $SEEDS; do
    python -m tasks.bio_experiments.train \
        --seed $SEED \
        --parity_length $PARITY_LENGTH \
        --epochs $EPOCHS \
        --output_dir $OUTPUT_DIR \
        --use_bio \
        --use_short_term_plasticity \
        --use_homeostasis \
        --use_lateral_inhibition \
        --use_refractory \
        --use_synaptic_noise \
        --experiment_name "full_bio_seed${SEED}"
done

# 3. Individual mechanism ablations
MECHANISMS=("use_short_term_plasticity" "use_homeostasis" "use_lateral_inhibition" "use_refractory" "use_synaptic_noise")

for MECH in "${MECHANISMS[@]}"; do
    for SEED in $SEEDS; do
        python -m tasks.bio_experiments.train \
            --seed $SEED \
            --parity_length $PARITY_LENGTH \
            --epochs $EPOCHS \
            --output_dir $OUTPUT_DIR \
            --use_bio \
            --${MECH} \
            --experiment_name "${MECH}_only_seed${SEED}"
    done
done

# 4. Ablations: Full bio minus one mechanism
for MECH in "${MECHANISMS[@]}"; do
    for SEED in $SEEDS; do
        # Build args excluding current mechanism
        ARGS=""
        for M in "${MECHANISMS[@]}"; do
            if [ "$M" != "$MECH" ]; then
                ARGS="$ARGS --$M"
            fi
        done
        
        python -m tasks.bio_experiments.train \
            --seed $SEED \
            --parity_length $PARITY_LENGTH \
            --epochs $EPOCHS \
            --output_dir $OUTPUT_DIR \
            --use_bio \
            $ARGS \
            --experiment_name "full_minus_${MECH}_seed${SEED}"
    done
done

echo "Ablation study complete!"