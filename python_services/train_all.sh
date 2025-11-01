#!/bin/bash
#
# train_all.sh - Train all 3 text-to-audio projection models sequentially
#
# This script trains the projection models in order:
# 1. Music2Latent (576D) - fastest, validates setup
# 2. MuQ (1536D) - medium complexity
# 3. MERT (76,800D) - largest, most challenging
#

set -e  # Exit on error

# Configuration
DATA_DIR="${DATA_DIR:-data/musicbench_embeddings}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-logs/tensorboard}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"

echo "========================================="
echo "Training Text-to-Audio Projection Models"
echo "========================================="
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Log directory: $LOG_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo ""

# Check if embeddings exist
if [ ! -f "$DATA_DIR/embeddings.h5" ]; then
    echo "ERROR: Embeddings file not found at $DATA_DIR/embeddings.h5"
    echo "Please run prepare_dataset.py first!"
    exit 1
fi

# Function to train a model
train_model() {
    local encoder=$1
    echo ""
    echo "========================================="
    echo "Training $encoder projection model"
    echo "========================================="
    echo "Start time: $(date)"

    python train.py \
        --audio-encoder "$encoder" \
        --data-dir "$DATA_DIR" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --log-dir "$LOG_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE"

    echo "Finished $encoder at: $(date)"
    echo ""
}

# Train models in order of increasing complexity
echo "Starting training sequence..."
START_TIME=$(date +%s)

# 1. Music2Latent (smallest, fastest)
train_model "latent"

# 2. MuQ (medium)
train_model "muq"

# 3. MERT (largest)
train_model "mert"

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "========================================="
echo "All models trained successfully!"
echo "========================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Checkpoints saved in: $CHECKPOINT_DIR"
echo "  - latent_best_r1.pt"
echo "  - muq_best_r1.pt"
echo "  - mert_best_r1.pt"
echo ""
echo "TensorBoard logs in: $LOG_DIR"
echo "  View with: tensorboard --logdir $LOG_DIR"
echo ""

# Create summary
SUMMARY_FILE="$CHECKPOINT_DIR/training_summary.json"
echo "Creating training summary..."

python - <<EOF
import json
import os

summary = {}
for encoder in ['latent', 'muq', 'mert']:
    metrics_file = os.path.join('$CHECKPOINT_DIR', f'{encoder}_test_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            summary[encoder] = json.load(f)

with open('$SUMMARY_FILE', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nTraining summary saved to: $SUMMARY_FILE")
print("\nFinal Test Results:")
print("-" * 60)
for encoder, metrics in summary.items():
    print(f"\n{encoder.upper()}:")
    print(f"  Text→Audio R@1:  {metrics.get('t2a_R@1', 0):.2f}%")
    print(f"  Text→Audio R@5:  {metrics.get('t2a_R@5', 0):.2f}%")
    print(f"  Text→Audio R@10: {metrics.get('t2a_R@10', 0):.2f}%")
    print(f"  Text→Audio MRR:  {metrics.get('t2a_MRR', 0):.4f}")
EOF

echo ""
echo "Training complete! 🎉"
