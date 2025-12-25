#!/bin/bash

for SEED in 42 43 44 45 46 47 48 49 50 51; do
    echo "Running experiment with seed: $SEED"
    
    if [ -d "splits" ]; then
        rm -rf splits
    fi

    export RANDOM_SEED=$SEED    
    python generate_split.py
    
    export EXP_NAME="base_seed_${SEED}"
    export SUBMISSION_PATH="${EXP_NAME}.csv"

    python train_simple_mask_model_base.py >> "logs/training_base_seed_${SEED}.log"
    python run_on_test_base.py >> "logs/run_on_test_base_seed_${SEED}.log"
    python score.py >> "logs/score_base_seed_${SEED}.log"

    export EXP_NAME="pca_seed_${SEED}"
    export SUBMISSION_PATH="${EXP_NAME}.csv"

    python train_simple_mask_model_pca.py >> "logs/training_pca_seed_${SEED}.log"
    python run_on_test_pca.py >> "logs/run_on_test_pca_seed_${SEED}.log"
    python score.py >> "logs/score_pca_seed_${SEED}.log"
    
    echo "Completed experiment with seed: $SEED"
    echo "----------------------------------------"
done
