for SEED in 42 43 44 45 46 47 48 49 50 51; do
    echo "Running validation with seed: $SEED"
    
    export RANDOM_SEED=$SEED
    export EXP_NAME="simple_masking_seed_${SEED}"
    export SUBMISSION_PATH="${EXP_NAME}.csv"
    export DEVICE="cuda:0"
    export SPLITS_DIR="splits_balanced"

    python train_simple_mask_model.py >> "logs/run_validation_seed_${SEED}.log"
    python run_on_test.py >> "logs/run_validation_seed_${SEED}.log"
    python score.py >> "logs/run_validation_seed_${SEED}.log"

done
