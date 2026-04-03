#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <gpu_ids> <data_dir> <load_dir> <master_port> [ckpt_step_or_latest]"
  exit 1
fi

GPU_IDS="$1"
DATA_DIR="$2"
LOAD_DIR="$3"
MASTER_PORT="$4"
CKPT_STEP="${5:-latest}"

if [[ ! -d "$LOAD_DIR/ckpts" ]]; then
  echo "Missing checkpoint directory: $LOAD_DIR/ckpts"
  exit 1
fi

if [[ "$CKPT_STEP" == "latest" ]]; then
  if ! ls "$LOAD_DIR"/ckpts/misc_*.pt >/dev/null 2>&1; then
    echo "No misc_step checkpoint found in $LOAD_DIR/ckpts"
    exit 1
  fi
else
  if [[ ! -f "$LOAD_DIR/ckpts/misc_step$(printf '%07d' "$CKPT_STEP").pt" ]]; then
    echo "Missing misc checkpoint for step $CKPT_STEP in $LOAD_DIR/ckpts"
    exit 1
  fi
fi

NUM_GPUS=$(python - "$GPU_IDS" <<'PY'
import sys
print(len(sys.argv[1].split(',')))
PY
)

TRELLIS_SKIP_INIT_SNAPSHOT=1 TRELLIS_SKIP_DATASET_SNAPSHOT=1 CUDA_VISIBLE_DEVICES="$GPU_IDS" \
/data/kmxu/miniconda3/envs/trellis/bin/python train.py \
  --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16_crrc_lora_bs4_split4.json \
  --data_dir "$DATA_DIR" \
  --output_dir "$LOAD_DIR" \
  --load_dir "$LOAD_DIR" \
  --ckpt "$CKPT_STEP" \
  --num_gpus "$NUM_GPUS" \
  --master_port "$MASTER_PORT"
