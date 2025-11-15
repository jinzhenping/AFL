import os
import time
import argparse
import json
import jsonlines
import numpy as np

from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.lastfm_dataset import LastfmDataset
from dataset.mind_dataset import MindDataset

from utils.regular_function import split_user_response, split_rec_reponse
from utils.rw_process import append_jsonl
from utils.agent import RecAgent, UserModelAgent
from utils.model import SASRec
finish_num = 0
total = 0
correct = 0
mrr_sum = 0.0
ndcg5_sum = 0.0
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--model_path', type=str, default=None, help='Path to SASRec model. For MIND dataset, defaults to ./SASRec.pth if not specified.')
    parser.add_argument('--prior_file', type=str, default='')
    parser.add_argument('--stage', type=str, default='test')
    parser.add_argument('--cans_num', type=int, default=20)
    parser.add_argument('--sep', type=str, default=', ')
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--output_file', type=str, required=True, help='Output file path for evaluation results (JSONL format)')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--max_retry_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=333)
    parser.add_argument('--mp', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument("--save_info",action="store_true")
    parser.add_argument("--save_rec_dir", type=str, default=None)
    parser.add_argument("--save_user_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None, help='GPU device ID to use (e.g., 0 or 1). If not specified, uses CUDA_VISIBLE_DEVICES or default cuda device.')
    return parser.parse_args()

def recommend(data, args):
    start_time = time.time()
    rec_agent = RecAgent(args,'prior_rec')
    user_agent = UserModelAgent(args,'prior_rec')
    flag = False
    epoch = 1
    rec_item = None
    new_data_list = []
    while flag == False and epoch <= args.max_epoch:
        #rec agent
        retry_count = 0
        max_retries = 10
        while retry_count < max_retries:
            rec_agent_response = rec_agent.act(data)
            if rec_agent_response is None:
                retry_count += 1
                print(f"[WARNING] RecAgent API returned None for data id {data['id']}, retry {retry_count}/{max_retries}")
                if retry_count >= max_retries:
                    print(f"[ERROR] RecAgent failed after {max_retries} retries for data id {data['id']}")
                    return new_data_list, 0, 0.0, 0.0, args
                continue
            rec_reason, rec_item = split_rec_reponse(rec_agent_response)
            if rec_item is not None:
                break
            retry_count += 1
            if retry_count >= max_retries:
                print(f"[ERROR] Failed to parse rec_item after {max_retries} retries for data id {data['id']}")
                return new_data_list, 0, 0.0, 0.0, args
        if args.max_epoch == 1:
            new_data = {'id':data['id'],'seq_name':data['seq_name'], 'cans_name':data['cans_name'], 'correct_answer':data['correct_answer'], 'epoch':epoch, 'rec_res':rec_agent_response, 'user_res':None,'prior_answer':data['prior_answer']}
            new_data_list.append(new_data)
            
            memory_info = {"epoch":epoch, "rec_reason":rec_reason, "rec_item":rec_item, "user_reason":None}
            rec_agent.update_memory(memory_info)
            user_agent.update_memory(memory_info)
            break
        
        #user agent
        retry_count = 0
        max_retries = 10
        while retry_count < max_retries:
            user_agent_response = user_agent.act(data,rec_reason, rec_item)
            if user_agent_response is None:
                retry_count += 1
                print(f"[WARNING] UserAgent API returned None for data id {data['id']}, retry {retry_count}/{max_retries}")
                if retry_count >= max_retries:
                    print(f"[ERROR] UserAgent failed after {max_retries} retries for data id {data['id']}")
                    flag = False  # Assume rejection
                    user_reason = "API request failed"
                    break
                continue
            user_reason, flag  = split_user_response(user_agent_response)
            if flag is not None:
                break
            retry_count += 1
            if retry_count >= max_retries:
                print(f"[ERROR] Failed to parse user response after {max_retries} retries for data id {data['id']}")
                flag = False  # Assume rejection
                user_reason = "Failed to parse response"
                break

        # save
        new_data = {'id':data['id'],'seq_name':data['seq_name'], 'cans_name':data['cans_name'], 'correct_answer':data['correct_answer'], 'epoch':epoch, 'rec_res':rec_agent_response, 'user_res':user_agent_response,'prior_answer':data['prior_answer']}
        new_data_list.append(new_data)

        memory_info = {"epoch":epoch, "rec_reason":rec_reason, "rec_item":rec_item, "user_reason":user_reason}
        rec_agent.update_memory(memory_info)
        user_agent.update_memory(memory_info)
        #update 
        if flag:
            break

        epoch += 1
    end_time = time.time()
    print("recommendation time = ", end_time - start_time)
    # save
    if args.save_info:
        rec_file_path = os.path.join(args.save_rec_dir, f"{data['id']}.jsonl")
        user_file_path = os.path.join(args.save_user_dir, f"{data['id']}.jsonl")
        rec_agent.save_memory(rec_file_path)
        user_agent.save_memory(user_file_path)
    # evaluate
    # Calculate metrics
    correct_answer = data['correct_answer'].lower().strip()
    rec_item_lower = rec_item.lower() if rec_item else ""
    
    # Hit@1: whether the recommended item matches the correct answer
    hit_at_1 = 1 if rec_item_lower == correct_answer else 0
    
    # Calculate MRR (Mean Reciprocal Rank)
    # MRR = 1/rank where rank is the position of the first relevant item
    # Since we only have one final recommendation, if it's correct, MRR = 1.0
    # Otherwise, we check if correct answer is in the candidate list
    mrr = 0.0
    if hit_at_1 == 1:
        mrr = 1.0  # Recommended item is correct, rank = 1
    else:
        # Check if correct answer is in candidate list
        candidates = [can.lower().strip() for can in data['cans_name']]
        if correct_answer in candidates:
            # Find the rank (1-indexed) of correct answer in candidate list
            rank = candidates.index(correct_answer) + 1
            mrr = 1.0 / rank
        # If correct answer not in candidates, MRR = 0.0
    
    # Calculate nDCG@5 (Normalized Discounted Cumulative Gain at 5)
    # We need to create a ranked list of recommendations
    # Since we have multiple epochs, we can use the final recommendation as rank 1
    # and check if correct answer appears in top 5 candidates
    candidates = [can.lower().strip() for can in data['cans_name']]
    
    # Create a ranked list: final recommendation is at position 1
    # Then fill with other candidates (excluding the recommended one)
    ranked_list = []
    if rec_item_lower:
        ranked_list.append(rec_item_lower)
    
    # Add other candidates that are not the recommended item
    for can in candidates:
        if can != rec_item_lower and can not in ranked_list:
            ranked_list.append(can)
    
    # Take top 5
    top5_ranked = ranked_list[:5]
    
    # Create relevance list (binary: 1 if correct, 0 otherwise)
    relevance_list = [1.0 if item == correct_answer else 0.0 for item in top5_ranked]
    
    # Calculate DCG@5
    dcg = 0.0
    for i, rel in enumerate(relevance_list):
        if rel > 0:
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0, we want log2(2) = 1
    
    # Calculate IDCG@5 (ideal DCG: correct answer at position 1)
    idcg = 1.0 / np.log2(2)  # Only one relevant item at position 1
    
    if idcg > 0:
        ndcg5 = dcg / idcg
    else:
        ndcg5 = 0.0
    
    return new_data_list, hit_at_1, mrr, ndcg5, args

def setcallback(x):
    global finish_num
    global total
    global correct
    global mrr_sum
    global ndcg5_sum

    data_list, hit_at_1, mrr, ndcg5, args = x
    for data in data_list:
        append_jsonl(args.output_file, data)
    finish_num += 1
    correct += hit_at_1
    mrr_sum += mrr
    ndcg5_sum += ndcg5
    
    print("==============")
    print(f"current Hit@1 = {correct} / {finish_num} = {correct/finish_num:.4f}")
    print(f"current MRR = {mrr_sum} / {finish_num} = {mrr_sum/finish_num:.4f}")
    print(f"current nDCG@5 = {ndcg5_sum} / {finish_num} = {ndcg5_sum/finish_num:.4f}")
    print(f"global Hit@1 = {correct} / {total} = {correct/total:.4f}")
    print(f"global MRR = {mrr_sum} / {total} = {mrr_sum/total:.4f}")
    print(f"global nDCG@5 = {ndcg5_sum} / {total} = {ndcg5_sum/total:.4f}")
    print("==============")

def main(args):
    if args.output_file is None:
        raise ValueError("--output_file is required. Please specify the output file path.")
    
    if args.save_info and args.save_rec_dir is not None and not os.path.exists(args.save_rec_dir):
        os.makedirs(args.save_rec_dir)
    if args.save_info and args.save_user_dir is not None and not os.path.exists(args.save_user_dir):
        os.makedirs(args.save_user_dir)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    if 'lastfm' in args.data_dir:
        dataset = LastfmDataset(args.data_dir, args.stage, args.cans_num, args.sep, True)
    elif 'mind' in args.data_dir.lower():
        dataset = MindDataset(args.data_dir, args.stage, args.cans_num, args.sep, True)
    else:
        raise ValueError(f"Unsupported dataset: {args.data_dir}")
    global total
    data_list = []
    print("Loading dataset...")
    for data in tqdm(dataset):
        data_list.append(data)
    print(f"Loaded {len(data_list)} samples from dataset")
    import pandas as pd
    print(f"Loading prior file: {args.prior_file}")
    prior_df = pd.read_csv(args.prior_file)
    print(f"Prior file columns: {prior_df.columns.tolist()}")
    print(f"Prior file shape: {prior_df.shape}")
    
    # Debug: Check first few rows
    if len(prior_df) > 0:
        print(f"First row sample: {prior_df.iloc[0].to_dict()}")
        print(f"First 'generate' value type: {type(prior_df.iloc[0]['generate'])}")
        print(f"First 'generate' value: {prior_df.iloc[0]['generate']}")
    
    prior_list = prior_df.to_dict('records')
    prior_dict = {}
    for data in prior_list:
        prior_dict[data['id']] = data

    merge_data_list = []
    print("Merging with prior data...")
    for data in data_list:
        if data['id'] not in prior_dict:
            print(f"[WARNING] Data id {data['id']} not found in prior_file, skipping...")
            continue
        prior_entry = prior_dict[data['id']]
        if 'generate' not in prior_entry:
            print(f"[ERROR] 'generate' key not found in prior_file for id {data['id']}. Available keys: {prior_entry.keys()}")
            continue
        generate = prior_entry['generate']
        # Ensure generate is a string, not a dict or other type
        if isinstance(generate, dict):
            print(f"[WARNING] 'generate' is a dict for id {data['id']}, converting to string")
            generate = str(generate)
        elif not isinstance(generate, str):
            generate = str(generate)
        merge_data = data.copy()
        merge_data['prior_answer'] = generate
        merge_data_list.append(merge_data)
    
    print(f"Total samples to process: {len(merge_data_list)}")
    if len(merge_data_list) == 0:
        print("[ERROR] No data to process after merging with prior_file")
        return
    
    pool = multiprocessing.Pool(args.mp)
    total = len(merge_data_list)
    print(f"Starting evaluation with {args.mp} processes...")
    
    def error_callback(e):
        import traceback
        print(f"[ERROR] Task failed with exception:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        traceback.print_exc()
    
    for data in tqdm(merge_data_list, desc="Submitting tasks"):
        pool.apply_async(func=recommend, args=(data, args), callback=setcallback, error_callback=error_callback)
    pool.close()
    print("Waiting for all tasks to complete...")
    pool.join()
    print("All tasks completed!")
    
    # Print final metrics
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {total}")
    print(f"Hit@1: {correct} / {total} = {correct/total:.4f}")
    print(f"MRR: {mrr_sum} / {total} = {mrr_sum/total:.4f}")
    print(f"nDCG@5: {ndcg5_sum} / {total} = {ndcg5_sum/total:.4f}")
    print("="*50)

if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    main(args)
    