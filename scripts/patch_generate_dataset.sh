#!/bin/bash
# Patch generate_dataset.py to support KEMEROVO custom path
# Run this on the server

set -e

TARGET_FILE="/home/Program/FRcnn/Coronary_Angiography_Detection-main/scripts/dataset_generation/generate_dataset.py"

echo "Patching generate_dataset.py to add KEMEROVO custom path..."

# Backup original file
cp "$TARGET_FILE" "${TARGET_FILE}.backup"
echo "✓ Backup created: ${TARGET_FILE}.backup"

# Check if patch already applied
if grep -q "Special path for KEMEROVO" "$TARGET_FILE"; then
    echo "⚠️  Patch already applied! Skipping."
    exit 0
fi

# Find the line number where we need to insert
LINE_NUM=$(grep -n 'root_dirs = {' "$TARGET_FILE" | head -1 | cut -d: -f1)

if [ -z "$LINE_NUM" ]; then
    echo "ERROR: Could not find insertion point in $TARGET_FILE"
    exit 1
fi

# Calculate insertion point (3 lines after root_dirs = {...})
INSERT_LINE=$((LINE_NUM + 3))

# Create the patch content
PATCH_CONTENT='
    # Special path for KEMEROVO dataset (Stenosis detection)
    if "KEMEROVO" in datasets_to_process:
        root_dirs["KEMEROVO"] = "/data/MEDDataset"
'

# Insert the patch
sed -i "${INSERT_LINE}a\\${PATCH_CONTENT}" "$TARGET_FILE"

echo "✓ Patch applied successfully!"
echo ""
echo "Verifying patch..."
if grep -q "Special path for KEMEROVO" "$TARGET_FILE"; then
    echo "✓ Verification passed!"
    echo ""
    echo "Now run:"
    echo "  cd /home/Program/FRcnn/Coronary_Angiography_Detection-main/scripts/dataset_generation"
    echo "  rm -rf /data/combined_stenosis/stenosis_detection/json/*.json"
    echo "  python generate_dataset.py --config cfg_dsgen_combined.yaml"
else
    echo "✗ Verification failed!"
    echo "Restoring backup..."
    mv "${TARGET_FILE}.backup" "$TARGET_FILE"
    exit 1
fi
