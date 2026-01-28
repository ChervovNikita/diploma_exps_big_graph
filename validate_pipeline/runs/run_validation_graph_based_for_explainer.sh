for SEED in 42; do
    echo "Running validation with seed: $SEED"
    
    export RANDOM_SEED=$SEED
    export EXP_NAME="simple_masking_graph_based_seed_for_explainer_${SEED}"
    export SUBMISSION_PATH="${EXP_NAME}.csv"
    export DEVICE="cuda:0"
    export SPLITS_DIR="splits_for_graph_based_balanced_for_explainer"

    mkdir -p "$SPLITS_DIR"
    mkdir -p "logs"
    mkdir -p "checkpoints"

    python generate_split.py >> "logs/run_validation_graph_based_for_explainer_seed_${SEED}.log"
    python train_simple_mask_model_graph_based.py >> "logs/run_validation_graph_based_for_explainer_seed_${SEED}.log"
    python run_on_test_graph_based.py >> "logs/run_validation_graph_based_for_explainer_seed_${SEED}.log"
    python score.py >> "logs/run_validation_graph_based_for_explainer_seed_${SEED}.log"

done
