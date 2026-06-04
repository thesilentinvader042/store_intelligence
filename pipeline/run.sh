#!/bin/bash
# Brigade Road Bangalore — CCTV Detection Pipeline
#
# Usage:
#   ./pipeline/run.sh                          # uses defaults from .env
#   ./pipeline/run.sh ./clips ST1008 CAM_ENTRY
#
# The store has 4 CCTV cameras. Run this script once per camera,
# or use the --camera_id argument to process each feed separately.
# Events from all cameras are appended to the same events.jsonl file.
#
# Camera IDs (from store layout):
#   CAM_ENTRY        — entry glass door (far left)
#   CAM_FLOOR_FRONT  — main floor front half overhead
#   CAM_FLOOR_REAR   — back shelf row overhead
#   CAM_CASH         — cash counter / billing area

set -e

CLIP_DIR="${1:-${CLIP_DIR:-./clips}}"
STORE_ID="${2:-${STORE_ID:-ST1008}}"
CAMERA_ID="${3:-}"
LAYOUT="${LAYOUT_PATH:-./store_layout.json}"
OUTPUT="${OUTPUT_EVENTS:-./events.jsonl}"

echo "=== Brigade Road Bangalore — Detection Pipeline ==="
echo "  Store:    $STORE_ID  (Brigade_Bangalore)"
echo "  Clip dir: $CLIP_DIR"
echo "  Layout:   $LAYOUT"
echo "  Output:   $OUTPUT"
echo ""

# If a specific camera is given, process only that subfolder
if [ -n "$CAMERA_ID" ]; then
  echo "  Camera: $CAMERA_ID"
  python pipeline/detect.py \
    --clip_dir  "$CLIP_DIR/$CAMERA_ID" \
    --layout    "$LAYOUT" \
    --output    "$OUTPUT" \
    --store_id  "$STORE_ID" \
    --camera_id "$CAMERA_ID"
else
  # Process all 4 camera folders if they exist, else treat clip_dir as flat
  CAMERAS=("CAM_ENTRY" "CAM_FLOOR_FRONT" "CAM_FLOOR_REAR" "CAM_CASH")
  HAS_SUBDIRS=false
  for CAM in "${CAMERAS[@]}"; do
    if [ -d "$CLIP_DIR/$CAM" ]; then
      HAS_SUBDIRS=true
      break
    fi
  done

  if $HAS_SUBDIRS; then
    for CAM in "${CAMERAS[@]}"; do
      if [ -d "$CLIP_DIR/$CAM" ]; then
        echo "--- Processing camera: $CAM ---"
        python pipeline/detect.py \
          --clip_dir  "$CLIP_DIR/$CAM" \
          --layout    "$LAYOUT" \
          --output    "$OUTPUT" \
          --store_id  "$STORE_ID" \
          --camera_id "$CAM"
      fi
    done
  else
    # Flat directory — process all clips, default camera CAM_FLOOR_FRONT
    echo "  Camera: CAM_FLOOR_FRONT (default — flat clip dir)"
    python pipeline/detect.py \
      --clip_dir  "$CLIP_DIR" \
      --layout    "$LAYOUT" \
      --output    "$OUTPUT" \
      --store_id  "$STORE_ID" \
      --camera_id "CAM_FLOOR_FRONT"
  fi
fi

echo ""
echo "Done. Events written to: $OUTPUT"
echo "Total event count: $(wc -l < "$OUTPUT" 2>/dev/null || echo 0)"
