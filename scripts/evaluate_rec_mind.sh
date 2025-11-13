#!/bin/bash
# Evaluate recommendations for MIND dataset

work_dir=''
cd $work_dir

# GPU 설정 (0 또는 1 등 원하는 GPU 번호 지정)
GPU_ID=1

# 기본 설정
STAGE="test"
CANS_NUM=5
MAX_EPOCH=5
MODEL="gpt-4o-mini" 
API_KEY="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"
MAX_RETRY_NUM=5
SEED=333
MP=16
TEMPERATURE=0.0
LABEL="mind_experiment"
P_MODEL='SASRec'

# MIND 데이터셋 설정
DATA_DIR="./mind"
PRIOR_FILE="./data/prior_mind/SASRec.csv"
MODEL_PATH="./SASRec.pth"  # 또는 모델 경로 지정

# OUTPUT_FILE은 다른 변수들이 정의된 후에 설정
OUTPUT_FILE="./results/mind_eval.jsonl"

SAVE_REC_DIR="./data/mind_rec_save/rec_${P_MODEL}_${LABEL}_${MODEL}_${MAX_EPOCH}"
SAVE_USER_DIR="./data/mind_rec_save/user_${P_MODEL}_${LABEL}_${MODEL}_${MAX_EPOCH}"

# OUTPUT_FILE이 비어있는지 확인
if [ -z "$OUTPUT_FILE" ]; then
    echo "Error: OUTPUT_FILE is empty. Please check the variable expansion."
    exit 1
fi

echo "DATA_DIR = ${DATA_DIR}"
echo "MODEL_PATH = ${MODEL_PATH}"
echo "PRIOR_FILE = ${PRIOR_FILE}"
echo "STAGE = ${STAGE}"
echo "CANS_NUM = ${CANS_NUM}"
echo "MAX_EPOCH = ${MAX_EPOCH}"
echo "MODEL = ${MODEL}"
echo "API_KEY = ${API_KEY:0:20}..."  # API 키 일부만 표시
echo "MAX_RETRY_NUM = ${MAX_RETRY_NUM}"
echo "SEED = ${SEED}"
echo "TEMPERATURE = ${TEMPERATURE}"
echo "MP = ${MP}"
echo "GPU_ID = ${GPU_ID}"
echo "OUTPUT_FILE = ${OUTPUT_FILE}"
echo "SAVE_REC_DIR = ${SAVE_REC_DIR}"
echo "SAVE_USER_DIR = ${SAVE_USER_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./afl/evaluate_rec.py \
    --data_dir="${DATA_DIR}" \
    --model_path="${MODEL_PATH}" \
    --prior_file="${PRIOR_FILE}" \
    --stage="${STAGE}" \
    --cans_num="${CANS_NUM}" \
    --max_epoch="${MAX_EPOCH}" \
    --output_file="${OUTPUT_FILE}" \
    --model="${MODEL}" \
    --api_key="${API_KEY}" \
    --max_retry_num="${MAX_RETRY_NUM}" \
    --seed="${SEED}" \
    --mp="${MP}" \
    --temperature="${TEMPERATURE}" \
    --gpu="${GPU_ID}" \
    --save_info \
    --save_rec_dir="${SAVE_REC_DIR}" \
    --save_user_dir="${SAVE_USER_DIR}"

