for SEED in 42 43 44 45 46 47 48 49 50 51; do
    echo "Running validation with seed: $SEED"
    
    export RANDOM_SEED=$SEED
    export EXP_NAME="simple_masking_graph_based_v2_seed_${SEED}"
    export SUBMISSION_PATH="${EXP_NAME}.csv"
    export VERSION="v2"
    export DEVICE="cuda:0"
    export SPLITS_DIR="splits_for_graph_based_v2_balanced"

    mkdir -p "$SPLITS_DIR"
    mkdir -p "logs"
    mkdir -p "checkpoints"

    python generate_split_v2.py >> "logs/run_validation_graph_based_v2_seed_${SEED}.log"
    python train_simple_mask_model_graph_based.py >> "logs/run_validation_graph_based_v2_seed_${SEED}.log"
    python run_on_test_graph_based.py >> "logs/run_validation_graph_based_v2_seed_${SEED}.log"
    python score.py >> "logs/run_validation_graph_based_v2_seed_${SEED}.log"

done
