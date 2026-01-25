for SEED in 42 43 44 45 46 47 48 49 50 51; do
    echo "Running validation with seed: $SEED"
    
    export RANDOM_SEED=$SEED
    export EXP_NAME="base_masking_v2_seed_${SEED}"
    export VERSION="v2"
    export SUBMISSION_PATH="${EXP_NAME}.csv"
    export DEVICE="cuda:1"
    export SPLITS_DIR="splits_for_base_v2_balanced"

    mkdir -p "$SPLITS_DIR"
    mkdir -p "logs"
    mkdir -p "checkpoints"

    python generate_split_v2.py >> "logs/run_validation_v2_seed_${SEED}.log"
    python train_base_mask_model.py >> "logs/run_validation_v2_seed_${SEED}.log"
    python run_on_test_base.py >> "logs/run_validation_v2_seed_${SEED}.log"
    python score.py >> "logs/run_validation_v2_seed_${SEED}.log"

done
