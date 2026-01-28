for SEED in 42 43 44 45 46; do
    echo "Running validation with seed: $SEED"
    
    export RANDOM_SEED=$SEED
    export EXP_NAME="simple_masking_tabm_v2_seed_${SEED}"
    export SUBMISSION_PATH="${EXP_NAME}.csv"
    export DEVICE="cuda:0"
    export VERSION="v2"
    export SPLITS_DIR="splits_for_tabm_v2_balanced_0"

    mkdir -p "$SPLITS_DIR"
    mkdir -p "logs"
    mkdir -p "checkpoints"

    python generate_split_v2.py >> "logs/run_validation_tabm_v2_seed_${SEED}.log"
    python train_tabm_mask_model.py >> "logs/run_validation_tabm_v2_seed_${SEED}.log"
    python run_on_test_tabm.py >> "logs/run_validation_tabm_v2_seed_${SEED}.log"
    python score.py >> "logs/run_validation_tabm_v2_seed_${SEED}.log"

done
