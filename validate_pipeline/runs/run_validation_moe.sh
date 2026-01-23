for SEED in 42 43 44 45 46 47 48 49 50 51; do
    echo "Running validation with seed: $SEED"
    
    export RANDOM_SEED=$SEED
    export EXP_NAME="moe_masking_seed_${SEED}"
    export SUBMISSION_PATH="${EXP_NAME}.csv"
    export DEVICE="cuda:0"
    export SPLITS_DIR="splits_for_moe_balanced"

    mkdir -p "$SPLITS_DIR"
    mkdir -p "logs"
    mkdir -p "checkpoints"

    python generate_split.py >> "logs/run_validation_moe_seed_${SEED}.log"
    python train_moe_mask_model.py >> "logs/run_validation_moe_seed_${SEED}.log"
    python run_on_test_moe.py >> "logs/run_validation_moe_seed_${SEED}.log"
    python score.py >> "logs/run_validation_moe_seed_${SEED}.log"

done
