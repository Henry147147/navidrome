#!/bin/bash
#
# Convenience script for running preprocessing
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DATASETS_DIR="../../data/datasets"
OUTPUT_DIR="../../data/preprocessed"
DEVICE="cuda"
BATCH_SIZE=1
MAX_SAMPLES=""
FORCE=""
DATASET=""
MODEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            echo -e "${YELLOW}Running in test mode (100 samples)${NC}"
            MAX_SAMPLES="--max-samples 100"
            shift
            ;;
        --force)
            FORCE="--force"
            shift
            ;;
        --cpu)
            echo -e "${YELLOW}Using CPU instead of CUDA${NC}"
            DEVICE="cpu"
            shift
            ;;
        --dataset)
            DATASET="--dataset $2"
            shift 2
            ;;
        --model)
            MODEL="--model $2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --test           Run in test mode (100 samples only)"
            echo "  --force          Force re-processing existing embeddings"
            echo "  --cpu            Use CPU instead of CUDA"
            echo "  --dataset NAME   Process specific dataset only"
            echo "  --model NAME     Use specific model only"
            echo "  --batch-size N   Set batch size (default: 8)"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Process all datasets with all models"
            echo "  $0 --test                       # Test run with 100 samples"
            echo "  $0 --dataset musiccaps          # Process only musiccaps"
            echo "  $0 --model muq                  # Use only muq model"
            echo "  $0 --force                      # Re-process everything"
            echo "  $0 --cpu --batch-size 4         # Use CPU with smaller batches"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Music Dataset Preprocessing${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Datasets dir: $DATASETS_DIR"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Device:       $DEVICE"
echo "  Batch size:   $BATCH_SIZE"
if [ -n "$DATASET" ]; then
    echo "  Dataset:      $(echo $DATASET | cut -d' ' -f2)"
fi
if [ -n "$MODEL" ]; then
    echo "  Model:        $(echo $MODEL | cut -d' ' -f2)"
fi
if [ -n "$MAX_SAMPLES" ]; then
    echo -e "  ${YELLOW}Test mode:    100 samples${NC}"
fi
if [ -n "$FORCE" ]; then
    echo -e "  ${YELLOW}Force mode:   Enabled${NC}"
fi
echo ""

# Check if datasets directory exists
if [ ! -d "$DATASETS_DIR" ]; then
    echo -e "${RED}Error: Datasets directory not found: $DATASETS_DIR${NC}"
    echo "Please download datasets first using:"
    echo "  cd ../datasets && python download_all.py --all"
    exit 1
fi

# Run preprocessing
echo -e "${GREEN}Starting preprocessing...${NC}"
echo ""

python preprocess_all.py \
    --datasets-dir "$DATASETS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    $DATASET \
    $MODEL \
    $MAX_SAMPLES \
    $FORCE

PREPROCESS_EXIT_CODE=$?

echo ""
if [ $PREPROCESS_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Preprocessing Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Run verification
    echo -e "${GREEN}Running verification...${NC}"
    echo ""
    python verify_embeddings.py --preprocessed-dir "$OUTPUT_DIR"

    echo ""
    echo -e "${GREEN}All done!${NC}"
    echo ""
    echo "Next steps:"
    echo "  - View detailed verification: python verify_embeddings.py --detailed"
    echo "  - Check summary: cat $OUTPUT_DIR/preprocessing_summary.json"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Preprocessing Failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check the logs above for error details."
    exit 1
fi
