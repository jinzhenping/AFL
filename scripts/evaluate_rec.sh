work_dir=''
cd $work_dir

# GPU 설정 (0 또는 1 등 원하는 GPU 번호 지정)
GPU_ID=0

# 기본 설정
STAGE="test"
CANS_NUM=20
MAX_EPOCH=5
MODEL="gpt-4o-mini-2024-07-18" 
API_KEY="" 
MAX_RETRY_NUM=5
SEED=303
MP=16
TEMPERATURE=0.0
LABEL="recommendation_experiment"
P_MODEL='SASRec'

# 데이터셋 설정 (lastfm 또는 mind)
# LastFM 사용 시:
DATA_DIR="./data/lastfm/"
PRIOR_FILE="./data/prior_lastfm/SASRec.csv"
MODEL_PATH="./output/lastfm/SASRec.pt"

# MIND 사용 시 아래 주석을 해제하고 위 설정을 주석 처리:
# DATA_DIR="./mind"
# PRIOR_FILE="./data/prior_mind/SASRec.csv"  # 이 파일을 먼저 생성해야 합니다
# MODEL_PATH="./SASRec.pth"  # 또는 모델 경로 지정
# CANS_NUM=5  # MIND는 보통 5개 후보

# OUTPUT_FILE은 다른 변수들이 정의된 후에 설정
OUTPUT_FILE="./data/lastfm_rec_eval/${P_MODEL}_${LABEL}_${MODEL}_${SEED}_${TEMPERATURE}_${MP}_${MAX_EPOCH}.jsonl"
# MIND 사용 시:
# OUTPUT_FILE="./data/mind_rec_eval/${P_MODEL}_${LABEL}_${MODEL}_${SEED}_${TEMPERATURE}_${MP}_${MAX_EPOCH}.jsonl"

SAVE_REC_DIR="./data/lastfm_rec_save/rec_${P_MODEL}_${LABEL}_${MODEL}_${MAX_EPOCH}"
SAVE_USER_DIR="./data/lastfm_rec_save/user_${P_MODEL}_${LABEL}_${MODEL}_${MAX_EPOCH}"

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
echo "API_KEY = ${API_KEY}"
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



    

