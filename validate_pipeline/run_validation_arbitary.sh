export SEED=42
export SPLITS_DIR="splits_for_simple_arbitary_balanced"
# python generate_split.py >> "logs/run_validation_seed_arbitary_${SEED}.log"

for MODEL_TYPE in TAGConv GCNConv SGConv SAGEConv LEConv; do
    echo "Running validation with seed: $SEED"
    
    export RANDOM_SEED=$SEED
    export EXP_NAME="simple_masking_seed_${MODEL_TYPE}_${SEED}"
    export SUBMISSION_PATH="${EXP_NAME}.csv"
    export DEVICE="cuda:1"
    export MODEL_TYPE=$MODEL_TYPE

    mkdir -p "$SPLITS_DIR"
    mkdir -p "logs"
    mkdir -p "checkpoints"

    echo "Running validation with seed: $SEED and model type: $MODEL_TYPE" >> "logs/run_validation_seed_arbitary_${SEED}.log"

    # python train_simple_arbitary_mask_model.py >> "logs/run_validation_seed_arbitary_${SEED}.log"
    python run_on_test_arbitary.py >> "logs/run_validation_seed_arbitary_${SEED}.log"
    python score.py >> "logs/run_validation_seed_arbitary_${SEED}.log"

done
