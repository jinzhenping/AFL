#!/bin/bash
# Generate prior recommendations for MIND dataset using SASRec model

work_dir=''
cd $work_dir

# GPU 설정 (0 또는 1 등 원하는 GPU 번호 지정)
GPU_ID=0

# 데이터셋 및 모델 설정
DATA_DIR="./mind"
MODEL_PATH="./SASRec.pth"  # 또는 모델 경로 지정
OUTPUT_FILE="./data/prior_mind/SASRec.csv"
STAGE="test"
CANS_NUM=5

echo "Generating prior recommendations for MIND dataset..."
echo "DATA_DIR = ${DATA_DIR}"
echo "MODEL_PATH = ${MODEL_PATH}"
echo "OUTPUT_FILE = ${OUTPUT_FILE}"
echo "STAGE = ${STAGE}"
echo "CANS_NUM = ${CANS_NUM}"
echo "GPU_ID = ${GPU_ID}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./afl/generate_prior.py \
    --data_dir="${DATA_DIR}" \
    --model_path="${MODEL_PATH}" \
    --output_file="${OUTPUT_FILE}" \
    --stage="${STAGE}" \
    --cans_num="${CANS_NUM}" \
    --gpu="${GPU_ID}"

echo "Done! Prior file saved to ${OUTPUT_FILE}"

