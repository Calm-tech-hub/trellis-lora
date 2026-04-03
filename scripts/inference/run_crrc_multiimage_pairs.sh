#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="$ROOT_DIR/scripts/inference/example_lora.py"

STAGE1_CKPT="/date3/nju/3dgen_train/outputs/crrc_ss_flow_lora_from_base/ckpts/denoiser_ema0.9999_step0110000.pt"
STAGE2_CKPT="/date3/nju/3dgen_train/outputs/crrc_slat_flow_lora_from_base/ckpts/denoiser_ema0.9999_step0100000.pt"
OUTPUT_ROOT="/date3/nju/3dgen_train/outputs"

COMMON_ARGS=(
  --seed 1
  --ss-steps 12
  --ss-cfg-strength 7.5
  --slat-steps 12
  --slat-cfg-strength 3
  --multiimage-mode multidiffusion
)

run_base_base() {
  local case_name="$1"
  local pair_name="$2"
  local front_image="$3"
  local side_image="$4"

  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --images "$front_image" "$side_image" \
    --output-dir "$OUTPUT_ROOT/${case_name}_${pair_name}_base_base" \
    "${COMMON_ARGS[@]}"
}

run_lora_lora() {
  local case_name="$1"
  local pair_name="$2"
  local front_image="$3"
  local side_image="$4"

  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --images "$front_image" "$side_image" \
    --stage1-ckpt "$STAGE1_CKPT" \
    --stage2-ckpt "$STAGE2_CKPT" \
    --output-dir "$OUTPUT_ROOT/${case_name}_${pair_name}_lora_lora" \
    "${COMMON_ARGS[@]}"
}

# CJ1-01
run_base_base "CJ1_01" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/a5101e9c6b9f10d4ef8f1d036714a34e53e950861fca6c8e97a8903c940906f2/043.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/a5101e9c6b9f10d4ef8f1d036714a34e53e950861fca6c8e97a8903c940906f2/049.png"
run_lora_lora "CJ1_01" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/a5101e9c6b9f10d4ef8f1d036714a34e53e950861fca6c8e97a8903c940906f2/043.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/a5101e9c6b9f10d4ef8f1d036714a34e53e950861fca6c8e97a8903c940906f2/049.png"
run_base_base "CJ1_01" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/a5101e9c6b9f10d4ef8f1d036714a34e53e950861fca6c8e97a8903c940906f2/043.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/a5101e9c6b9f10d4ef8f1d036714a34e53e950861fca6c8e97a8903c940906f2/048.png"
run_lora_lora "CJ1_01" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/a5101e9c6b9f10d4ef8f1d036714a34e53e950861fca6c8e97a8903c940906f2/043.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/a5101e9c6b9f10d4ef8f1d036714a34e53e950861fca6c8e97a8903c940906f2/048.png"

# CR200J-D-01
run_base_base "CR200J_D_01" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/ad7cb72de303439fb31da5047125fcd633bc30afa33b0e3fd8695fc289ec17ec/058.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/ad7cb72de303439fb31da5047125fcd633bc30afa33b0e3fd8695fc289ec17ec/025.png"
run_lora_lora "CR200J_D_01" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/ad7cb72de303439fb31da5047125fcd633bc30afa33b0e3fd8695fc289ec17ec/058.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/ad7cb72de303439fb31da5047125fcd633bc30afa33b0e3fd8695fc289ec17ec/025.png"
run_base_base "CR200J_D_01" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/ad7cb72de303439fb31da5047125fcd633bc30afa33b0e3fd8695fc289ec17ec/058.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/ad7cb72de303439fb31da5047125fcd633bc30afa33b0e3fd8695fc289ec17ec/024.png"
run_lora_lora "CR200J_D_01" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/ad7cb72de303439fb31da5047125fcd633bc30afa33b0e3fd8695fc289ec17ec/058.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/ad7cb72de303439fb31da5047125fcd633bc30afa33b0e3fd8695fc289ec17ec/024.png"

# CRH2C-008
run_base_base "CRH2C_008" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/3ac2da6b36206dcfcd135d58067f26fe198c3f664cfdeb3f7f5f6a9575375df8/033.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/3ac2da6b36206dcfcd135d58067f26fe198c3f664cfdeb3f7f5f6a9575375df8/028.png"
run_lora_lora "CRH2C_008" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/3ac2da6b36206dcfcd135d58067f26fe198c3f664cfdeb3f7f5f6a9575375df8/033.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/3ac2da6b36206dcfcd135d58067f26fe198c3f664cfdeb3f7f5f6a9575375df8/028.png"
run_base_base "CRH2C_008" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/3ac2da6b36206dcfcd135d58067f26fe198c3f664cfdeb3f7f5f6a9575375df8/033.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/3ac2da6b36206dcfcd135d58067f26fe198c3f664cfdeb3f7f5f6a9575375df8/029.png"
run_lora_lora "CRH2C_008" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/3ac2da6b36206dcfcd135d58067f26fe198c3f664cfdeb3f7f5f6a9575375df8/033.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/3ac2da6b36206dcfcd135d58067f26fe198c3f664cfdeb3f7f5f6a9575375df8/029.png"

# CR200J1-01
run_base_base "CR200J1_01" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/56dc531d3e8cd687fc653ef87972425f1d03ebb693217d5bacb80de2daa369c4/056.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/56dc531d3e8cd687fc653ef87972425f1d03ebb693217d5bacb80de2daa369c4/027.png"
run_lora_lora "CR200J1_01" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/56dc531d3e8cd687fc653ef87972425f1d03ebb693217d5bacb80de2daa369c4/056.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/56dc531d3e8cd687fc653ef87972425f1d03ebb693217d5bacb80de2daa369c4/027.png"
run_base_base "CR200J1_01" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/56dc531d3e8cd687fc653ef87972425f1d03ebb693217d5bacb80de2daa369c4/056.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/56dc531d3e8cd687fc653ef87972425f1d03ebb693217d5bacb80de2daa369c4/026.png"
run_lora_lora "CR200J1_01" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/56dc531d3e8cd687fc653ef87972425f1d03ebb693217d5bacb80de2daa369c4/056.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/56dc531d3e8cd687fc653ef87972425f1d03ebb693217d5bacb80de2daa369c4/026.png"

# CR200J2-JT001_02
run_base_base "CR200J2_JT001_02" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/f98fa2bf5c62a75f184506d732eb265a9d2070482c20c51e6856d7bd2bf4d6c4/055.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/f98fa2bf5c62a75f184506d732eb265a9d2070482c20c51e6856d7bd2bf4d6c4/029.png"
run_lora_lora "CR200J2_JT001_02" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/f98fa2bf5c62a75f184506d732eb265a9d2070482c20c51e6856d7bd2bf4d6c4/055.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/f98fa2bf5c62a75f184506d732eb265a9d2070482c20c51e6856d7bd2bf4d6c4/029.png"
run_base_base "CR200J2_JT001_02" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/f98fa2bf5c62a75f184506d732eb265a9d2070482c20c51e6856d7bd2bf4d6c4/055.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/f98fa2bf5c62a75f184506d732eb265a9d2070482c20c51e6856d7bd2bf4d6c4/028.png"
run_lora_lora "CR200J2_JT001_02" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/f98fa2bf5c62a75f184506d732eb265a9d2070482c20c51e6856d7bd2bf4d6c4/055.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/f98fa2bf5c62a75f184506d732eb265a9d2070482c20c51e6856d7bd2bf4d6c4/028.png"

# CR450AF_01_
run_base_base "CR450AF_01_" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/250e3d40dee33634e2509b394a262d328b5dbf4513ea08946531ffcdce38198d/058.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/250e3d40dee33634e2509b394a262d328b5dbf4513ea08946531ffcdce38198d/036.png"
run_lora_lora "CR450AF_01_" "front_right" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/250e3d40dee33634e2509b394a262d328b5dbf4513ea08946531ffcdce38198d/058.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/250e3d40dee33634e2509b394a262d328b5dbf4513ea08946531ffcdce38198d/036.png"
run_base_base "CR450AF_01_" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/250e3d40dee33634e2509b394a262d328b5dbf4513ea08946531ffcdce38198d/058.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/250e3d40dee33634e2509b394a262d328b5dbf4513ea08946531ffcdce38198d/037.png"
run_lora_lora "CR450AF_01_" "front_left" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/250e3d40dee33634e2509b394a262d328b5dbf4513ea08946531ffcdce38198d/058.png" \
  "/date3/nju/3dgen_train/datasets/crrc/renders/250e3d40dee33634e2509b394a262d328b5dbf4513ea08946531ffcdce38198d/037.png"
