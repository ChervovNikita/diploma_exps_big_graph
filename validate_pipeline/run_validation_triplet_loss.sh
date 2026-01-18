for SEED in 42 43 44 45 46 47 48 49 50 51; do
    echo "Running validation with seed: $SEED"
    
    export RANDOM_SEED=$SEED
    export EXP_NAME="triplet_loss_seed_${SEED}"
    export SUBMISSION_PATH="${EXP_NAME}.csv"
    export DEVICE="cuda:1"
    export SPLITS_DIR="splits_for_triplet_loss_balanced"

    mkdir -p "$SPLITS_DIR"
    mkdir -p "logs"
    mkdir -p "checkpoints"

    python generate_split.py >> "logs/run_validation_triplet_loss_seed_${SEED}.log"
    python train_triplet_loss.py >> "logs/run_validation_triplet_loss_seed_${SEED}.log"
    python run_on_test_triplet_loss.py >> "logs/run_validation_triplet_loss_seed_${SEED}.log"
    python score.py >> "logs/run_validation_triplet_loss_seed_${SEED}.log"

done
