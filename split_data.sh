#!/bin/bash

# --- Configuration ---
# This is the folder containing the raw class folders (lung_aca, lung_n, lung_scc)
SOURCE_DIR="lung_image_sets"
# This is the target directory where the split data will be created (matches your python script)
TARGET_DIR="data_splits"

# Define the split ratios (Train / Validation / Test)
TRAIN_RATIO=0.8
VALID_RATIO=0.1
# Test ratio is implicitly 1.0 - TRAIN_RATIO - VALID_RATIO

# --- Setup and Validation ---

echo "Starting data split (80% Train, 10% Valid, 10% Test)..."

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' not found."
    echo "Please ensure your raw data (lung_aca, lung_n, lung_scc) is inside a folder named '$SOURCE_DIR'."
    exit 1
fi

# Create target directories (and clean up if they exist)
if [ -d "$TARGET_DIR" ]; then
    echo "Warning: Clearing existing '$TARGET_DIR' contents..."
    rm -rf "$TARGET_DIR"
fi

mkdir -p "$TARGET_DIR/train" "$TARGET_DIR/valid" "$TARGET_DIR/test"

# --- Splitting Logic ---

# Iterate over each class directory (e.g., lung_aca, lung_n, lung_scc)
for CLASS_DIR in "$SOURCE_DIR"/*/; do
    # Extract just the class name (e.g., lung_aca)
    CLASS_NAME=$(basename "$CLASS_DIR")
    
    echo "Processing class: $CLASS_NAME"
    
    # Create class subdirectories in the target split folders
    mkdir -p "$TARGET_DIR/train/$CLASS_NAME"
    mkdir -p "$TARGET_DIR/valid/$CLASS_NAME"
    mkdir -p "$TARGET_DIR/test/$CLASS_NAME"
    
    # Get list of all images in the class, randomly shuffled
    IMAGES=($(find "$CLASS_DIR" -maxdepth 1 -type f -print0 | xargs -0 shuf -e))
    TOTAL_COUNT=${#IMAGES[@]}
    
    if [ "$TOTAL_COUNT" -eq 0 ]; then
        echo "   Skipping: No images found in $CLASS_NAME."
        continue
    fi
    
    # Calculate split counts
    TRAIN_COUNT=$(printf "%.0f\n" $(echo "$TOTAL_COUNT * $TRAIN_RATIO" | bc))
    VALID_COUNT=$(printf "%.0f\n" $(echo "$TOTAL_COUNT * $VALID_RATIO" | bc))
    # Test count is the remainder
    TEST_COUNT=$((TOTAL_COUNT - TRAIN_COUNT - VALID_COUNT))

    if [ "$TEST_COUNT" -lt 0 ]; then
        TEST_COUNT=0
    fi
    
    echo "   Total: $TOTAL_COUNT | Train: $TRAIN_COUNT | Valid: $VALID_COUNT | Test: $TEST_COUNT"
    
    # 1. Copy (or link) files for TRAIN
    TRAIN_END=$((TRAIN_COUNT - 1))
    for i in $(seq 0 $TRAIN_END); do
        cp "${IMAGES[$i]}" "$TARGET_DIR/train/$CLASS_NAME/"
    done
    
    # 2. Copy (or link) files for VALIDATION
    VALID_START=$((TRAIN_COUNT))
    VALID_END=$((TRAIN_COUNT + VALID_COUNT - 1))
    for i in $(seq $VALID_START $VALID_END); do
        cp "${IMAGES[$i]}" "$TARGET_DIR/valid/$CLASS_NAME/"
    done
    
    # 3. Copy (or link) files for TEST
    TEST_START=$((TRAIN_COUNT + VALID_COUNT))
    TEST_END=$((TOTAL_COUNT - 1))
    for i in $(seq $TEST_START $TEST_END); do
        cp "${IMAGES[$i]}" "$TARGET_DIR/test/$CLASS_NAME/"
    done

done

echo "---"
echo "Data splitting complete! Your PyTorch script is now configured to use the '$TARGET_DIR' folder."
echo "Total classes processed: $(ls -d $TARGET_DIR/train/*/ | wc -l)"
