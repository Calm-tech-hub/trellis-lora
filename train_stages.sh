#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Core runtime options
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
DATA_DIRS="${DATA_DIRS:-}"
NUM_GPUS="${NUM_GPUS:--1}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12345}"
AUTO_RETRY="${AUTO_RETRY:-3}"
ATTN_BACKEND="${ATTN_BACKEND:-flash-attn}"
SPCONV_ALGO="${SPCONV_ALGO:-native}"
EXP_TAG="${EXP_TAG:-}"
LOAD_DIR="${LOAD_DIR:-}"
CKPT="${CKPT:-latest}"
TRYRUN="${TRYRUN:-0}"
PROFILE="${PROFILE:-0}"

# Dataset toolkit options
SUBSET="${SUBSET:-ABO}"
TOOLKIT_OUTPUT_DIR="${TOOLKIT_OUTPUT_DIR:-}"
TOOLKIT_RANK="${TOOLKIT_RANK:-0}"
TOOLKIT_WORLD_SIZE="${TOOLKIT_WORLD_SIZE:-1}"
NUM_VIEWS="${NUM_VIEWS:-150}"
COND_NUM_VIEWS="${COND_NUM_VIEWS:-24}"
MODEL_ROOT="${MODEL_ROOT:-$OUTPUT_ROOT}"
SS_ENC_PRETRAINED="${SS_ENC_PRETRAINED:-microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16}"
SLAT_ENC_PRETRAINED="${SLAT_ENC_PRETRAINED:-microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16}"
FEAT_MODEL="${FEAT_MODEL:-dinov2_vitl14_reg}"
ENC_MODEL="${ENC_MODEL:-}"
ENC_CKPT="${ENC_CKPT:-}"

# Config paths; point these to copied configs if you customize latent_model/pretrained paths.
SS_VAE_CONFIG="${SS_VAE_CONFIG:-configs/vae/ss_vae_conv3d_16l8_fp16.json}"
SLAT_VAE_GS_CONFIG="${SLAT_VAE_GS_CONFIG:-configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json}"
SLAT_VAE_RF_CONFIG="${SLAT_VAE_RF_CONFIG:-configs/vae/slat_vae_dec_rf_swin8_B_64l8_fp16.json}"
SLAT_VAE_MESH_CONFIG="${SLAT_VAE_MESH_CONFIG:-configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json}"
SS_FLOW_IMG_CONFIG="${SS_FLOW_IMG_CONFIG:-configs/generation/ss_flow_img_dit_L_16l8_fp16.json}"
SLAT_FLOW_IMG_CONFIG="${SLAT_FLOW_IMG_CONFIG:-configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json}"
SS_FLOW_TXT_B_CONFIG="${SS_FLOW_TXT_B_CONFIG:-configs/generation/ss_flow_txt_dit_B_16l8_fp16.json}"
SLAT_FLOW_TXT_B_CONFIG="${SLAT_FLOW_TXT_B_CONFIG:-configs/generation/slat_flow_txt_dit_B_64l8p2_fp16.json}"
SS_FLOW_TXT_L_CONFIG="${SS_FLOW_TXT_L_CONFIG:-configs/generation/ss_flow_txt_dit_L_16l8_fp16.json}"
SLAT_FLOW_TXT_L_CONFIG="${SLAT_FLOW_TXT_L_CONFIG:-configs/generation/slat_flow_txt_dit_L_64l8p2_fp16.json}"
SS_FLOW_TXT_XL_CONFIG="${SS_FLOW_TXT_XL_CONFIG:-configs/generation/ss_flow_txt_dit_XL_16l8_fp16.json}"
SLAT_FLOW_TXT_XL_CONFIG="${SLAT_FLOW_TXT_XL_CONFIG:-configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16.json}"

usage() {
    cat <<'EOF'
Usage:
  bash train_stages.sh <stage>

Training stages:
  ss_vae             Train sparse-structure VAE encoder/decoder
  slat_vae_gs        Train SLAT encoder + Gaussian decoder
  slat_vae_rf        Train Radiance Field decoder on SLAT latents
  slat_vae_mesh      Train Mesh decoder on SLAT latents
  ss_flow_img        Train image-conditioned stage-1 flow
  slat_flow_img      Train image-conditioned stage-2 flow
  ss_flow_txt_b      Train text-conditioned base stage-1 flow
  slat_flow_txt_b    Train text-conditioned base stage-2 flow
  ss_flow_txt_l      Train text-conditioned large stage-1 flow
  slat_flow_txt_l    Train text-conditioned large stage-2 flow
  ss_flow_txt_xl     Train text-conditioned xlarge stage-1 flow
  slat_flow_txt_xl   Train text-conditioned xlarge stage-2 flow
  all_image_core     Run the main image pipeline order reminder

Dataset-prep helpers:
  build_metadata     Refresh metadata.csv for one dataset folder
  render_cond        Render image conditions for image-conditioned training
  encode_ss_latent   Encode sparse-structure latents
  encode_latent      Encode SLAT latents

Common env vars:
  DATA_DIRS=/path/a,/path/b        Training dataset roots, comma separated
  OUTPUT_ROOT=outputs/exp_root     Where checkpoints are written
  NUM_GPUS=4                       GPUs per node; -1 means all visible GPUs
  NUM_NODES=1 NODE_RANK=0          Multi-node settings
  MASTER_ADDR=host MASTER_PORT=12345
  EXP_TAG=myrun                    Suffix added to output directory names
  LOAD_DIR=/path/to/old_run        Optional resume source
  CKPT=latest                      Optional resume checkpoint step
  TRYRUN=1                         Validate config without starting training

Toolkit env vars:
  SUBSET=ABO
  TOOLKIT_OUTPUT_DIR=/path/to/dataset_root
  TOOLKIT_RANK=0 TOOLKIT_WORLD_SIZE=1
  ENC_MODEL=slat_vae_enc_dec_gs_swin8_B_64l8_fp16 ENC_CKPT=100000
  MODEL_ROOT=outputs

Examples:
  DATA_DIRS=/data/abo bash train_stages.sh ss_vae
  DATA_DIRS=/data/abo,/data/objxl NUM_GPUS=8 bash train_stages.sh slat_flow_img
  TOOLKIT_OUTPUT_DIR=/data/abo bash train_stages.sh render_cond
  TOOLKIT_OUTPUT_DIR=/data/abo ENC_MODEL=ss_vae_conv3d_16l8_fp16 ENC_CKPT=100000 bash train_stages.sh encode_ss_latent
EOF
}

require_var() {
    local name="$1"
    local value="$2"
    if [[ -z "$value" ]]; then
        echo "[ERROR] Missing required env var: $name" >&2
        exit 1
    fi
}

append_common_train_args() {
    TRAIN_CMD+=(
        --num_nodes "$NUM_NODES"
        --node_rank "$NODE_RANK"
        --num_gpus "$NUM_GPUS"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
        --auto_retry "$AUTO_RETRY"
    )

    if [[ -n "$LOAD_DIR" ]]; then
        TRAIN_CMD+=(--load_dir "$LOAD_DIR" --ckpt "$CKPT")
    fi
    if [[ "$TRYRUN" == "1" ]]; then
        TRAIN_CMD+=(--tryrun)
    fi
    if [[ "$PROFILE" == "1" ]]; then
        TRAIN_CMD+=(--profile)
    fi
}

run_train_stage() {
    local stage_name="$1"
    local config_path="$2"
    require_var "DATA_DIRS" "$DATA_DIRS"

    export ATTN_BACKEND
    export SPCONV_ALGO

    local out_name="$stage_name"
    if [[ -n "$EXP_TAG" ]]; then
        out_name+="_$EXP_TAG"
    fi

    mkdir -p "$OUTPUT_ROOT"

    TRAIN_CMD=(
        "$PYTHON_BIN" train.py
        --config "$config_path"
        --output_dir "$OUTPUT_ROOT/$out_name"
        --data_dir "$DATA_DIRS"
    )
    append_common_train_args

    printf '[RUN] '
    printf '%q ' "${TRAIN_CMD[@]}"
    printf '\n'
    "${TRAIN_CMD[@]}"
}

run_toolkit_cmd() {
    require_var "TOOLKIT_OUTPUT_DIR" "$TOOLKIT_OUTPUT_DIR"

    printf '[RUN] '
    printf '%q ' "${TOOLKIT_CMD[@]}"
    printf '\n'
    "${TOOLKIT_CMD[@]}"
}

build_metadata() {
    TOOLKIT_CMD=(
        "$PYTHON_BIN" dataset_toolkits/build_metadata.py "$SUBSET"
        --output_dir "$TOOLKIT_OUTPUT_DIR"
    )
    run_toolkit_cmd
}

render_cond() {
    TOOLKIT_CMD=(
        "$PYTHON_BIN" dataset_toolkits/render_cond.py "$SUBSET"
        --output_dir "$TOOLKIT_OUTPUT_DIR"
        --num_views "$COND_NUM_VIEWS"
        --rank "$TOOLKIT_RANK"
        --world_size "$TOOLKIT_WORLD_SIZE"
    )
    run_toolkit_cmd
}

encode_ss_latent() {
    TOOLKIT_CMD=(
        "$PYTHON_BIN" dataset_toolkits/encode_ss_latent.py
        --output_dir "$TOOLKIT_OUTPUT_DIR"
        --rank "$TOOLKIT_RANK"
        --world_size "$TOOLKIT_WORLD_SIZE"
    )

    if [[ -n "$ENC_MODEL" ]]; then
        require_var "ENC_CKPT" "$ENC_CKPT"
        TOOLKIT_CMD+=(--enc_model "$ENC_MODEL" --ckpt "$ENC_CKPT" --model_root "$MODEL_ROOT")
    else
        TOOLKIT_CMD+=(--enc_pretrained "$SS_ENC_PRETRAINED")
    fi

    run_toolkit_cmd
}

encode_latent() {
    TOOLKIT_CMD=(
        "$PYTHON_BIN" dataset_toolkits/encode_latent.py
        --output_dir "$TOOLKIT_OUTPUT_DIR"
        --feat_model "$FEAT_MODEL"
        --rank "$TOOLKIT_RANK"
        --world_size "$TOOLKIT_WORLD_SIZE"
    )

    if [[ -n "$ENC_MODEL" ]]; then
        require_var "ENC_CKPT" "$ENC_CKPT"
        TOOLKIT_CMD+=(--enc_model "$ENC_MODEL" --ckpt "$ENC_CKPT" --model_root "$MODEL_ROOT")
    else
        TOOLKIT_CMD+=(--enc_pretrained "$SLAT_ENC_PRETRAINED")
    fi

    run_toolkit_cmd
}

print_all_image_core() {
    cat <<'EOF'
Recommended image-training order:
  1. Train sparse-structure VAE           -> bash train_stages.sh ss_vae
  2. Encode ss latents                    -> bash train_stages.sh encode_ss_latent
  3. Train SLAT VAE (enc + gs dec)        -> bash train_stages.sh slat_vae_gs
  4. Encode SLAT latents                  -> bash train_stages.sh encode_latent
  5. Optional RF decoder                  -> bash train_stages.sh slat_vae_rf
  6. Optional Mesh decoder                -> bash train_stages.sh slat_vae_mesh
  7. Render image conditions              -> bash train_stages.sh render_cond
  8. Train image stage-1 flow             -> bash train_stages.sh ss_flow_img
  9. Train image stage-2 flow             -> bash train_stages.sh slat_flow_img

Notes:
  - If you use pretrained encoders, steps 1-4 can be skipped.
  - If you train your own encoders, copy the JSON config and update dataset.args.latent_model.
  - The flow configs also reference pretrained decoders; update those paths in your copied configs if needed.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

case "$1" in
    ss_vae)
        run_train_stage "ss_vae_conv3d_16l8_fp16" "$SS_VAE_CONFIG"
        ;;
    slat_vae_gs)
        run_train_stage "slat_vae_enc_dec_gs_swin8_B_64l8_fp16" "$SLAT_VAE_GS_CONFIG"
        ;;
    slat_vae_rf)
        run_train_stage "slat_vae_dec_rf_swin8_B_64l8_fp16" "$SLAT_VAE_RF_CONFIG"
        ;;
    slat_vae_mesh)
        run_train_stage "slat_vae_dec_mesh_swin8_B_64l8_fp16" "$SLAT_VAE_MESH_CONFIG"
        ;;
    ss_flow_img)
        run_train_stage "ss_flow_img_dit_L_16l8_fp16" "$SS_FLOW_IMG_CONFIG"
        ;;
    slat_flow_img)
        run_train_stage "slat_flow_img_dit_L_64l8p2_fp16" "$SLAT_FLOW_IMG_CONFIG"
        ;;
    ss_flow_txt_b)
        run_train_stage "ss_flow_txt_dit_B_16l8_fp16" "$SS_FLOW_TXT_B_CONFIG"
        ;;
    slat_flow_txt_b)
        run_train_stage "slat_flow_txt_dit_B_64l8p2_fp16" "$SLAT_FLOW_TXT_B_CONFIG"
        ;;
    ss_flow_txt_l)
        run_train_stage "ss_flow_txt_dit_L_16l8_fp16" "$SS_FLOW_TXT_L_CONFIG"
        ;;
    slat_flow_txt_l)
        run_train_stage "slat_flow_txt_dit_L_64l8p2_fp16" "$SLAT_FLOW_TXT_L_CONFIG"
        ;;
    ss_flow_txt_xl)
        run_train_stage "ss_flow_txt_dit_XL_16l8_fp16" "$SS_FLOW_TXT_XL_CONFIG"
        ;;
    slat_flow_txt_xl)
        run_train_stage "slat_flow_txt_dit_XL_64l8p2_fp16" "$SLAT_FLOW_TXT_XL_CONFIG"
        ;;
    build_metadata)
        build_metadata
        ;;
    render_cond)
        render_cond
        ;;
    encode_ss_latent)
        encode_ss_latent
        ;;
    encode_latent)
        encode_latent
        ;;
    all_image_core)
        print_all_image_core
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown stage: $1" >&2
        usage
        exit 1
        ;;
esac
