for SEED in 42 43 44 45 46 47 48 49 50 51; do
    echo "Running validation with seed: $SEED"
    
    export RANDOM_SEED=$SEED
    export EXP_NAME="simple_masking_tabm_seed_${SEED}"
    export SUBMISSION_PATH="${EXP_NAME}.csv"
    export DEVICE="cuda:1"
    export SPLITS_DIR="splits_for_tabm_balanced"

    mkdir -p "$SPLITS_DIR"
    mkdir -p "logs"
    mkdir -p "checkpoints"

    python generate_split.py >> "logs/run_validation_tabm_seed_${SEED}.log"
    python train_tabm_mask_model.py >> "logs/run_validation_tabm_seed_${SEED}.log"
    python run_on_test_tabm.py >> "logs/run_validation_tabm_seed_${SEED}.log"
    python score.py >> "logs/run_validation_tabm_seed_${SEED}.log"

done
