# AFL

This is the code repository for the paper titled "Agentic Feedback Loop Modeling Improves Recommendation and User Simulation".


python afl/evaluate_rec.py \
    --data_dir=./mind \
    --output_file=./results/mind_eval.jsonl \  # 필수!
    --prior_file=./data/prior_mind/SASRec.csv \
    --stage=test \
    --cans_num=5 \
    --max_epoch=5 \
    --model=gpt-4o-mini \
    --api_key=YOUR_API_KEY \
    --mp=16 \
    --gpu=0
