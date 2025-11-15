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
invalid_users = 0

# Global model cache per process (for multiprocessing optimization)
_process_model_cache = None
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
    parser.add_argument("--max_users", type=int, default=None, help='Maximum number of users to evaluate. If not specified, evaluates all users in the dataset.')
    return parser.parse_args()

def recommend(data, args):
    global _process_model_cache
    
    start_time = time.time()
    rec_agent = RecAgent(args,'prior_rec')
    
    # Reuse model if available in this process (for multiprocessing optimization)
    # Each process loads the model once and reuses it for all users in that process
    if _process_model_cache is None:
        user_agent = UserModelAgent(args,'prior_rec')
        _process_model_cache = user_agent
        print(f"[INFO] Model loaded in process {os.getpid()}")
    else:
        # Reuse cached model but create new agent instance for memory isolation
        # Create a new agent but skip the expensive model loading
        user_agent = UserModelAgent.__new__(UserModelAgent)  # Create instance without calling __init__
        user_agent.memory = []
        user_agent.info_list = []
        user_agent.args = args
        user_agent.mode = 'prior_rec'
        user_agent.device = _process_model_cache.device
        user_agent.load_prompt()  # Still need prompts
        # Reuse cached model and mappings (copy all necessary attributes)
        user_agent.model = _process_model_cache.model
        user_agent.id2name = _process_model_cache.id2name
        user_agent.name2id = _process_model_cache.name2id
        # Copy model-related attributes that are needed
        if hasattr(_process_model_cache, 'item_num'):
            user_agent.item_num = _process_model_cache.item_num
        if hasattr(_process_model_cache, 'seq_size'):
            user_agent.seq_size = _process_model_cache.seq_size
        if hasattr(_process_model_cache, 'news_id_to_int'):
            user_agent.news_id_to_int = _process_model_cache.news_id_to_int
    flag = False
    epoch = 1
    rec_item = None
    rec_ranking = None
    new_data_list = []
    is_invalid_user = False
    while flag == False and epoch <= args.max_epoch:
        #rec agent
        retry_count = 0
        max_retries = 10
        api_start_time = time.time()
        max_api_time = 300  # 5 minutes timeout per API call attempt
        while retry_count < max_retries:
            # Check if we've been trying too long
            if time.time() - api_start_time > max_api_time:
                print(f"[ERROR] API call timeout after {max_api_time} seconds for data id {data['id']}")
                is_invalid_user = True
                if len(new_data_list) == 0:
                    new_data_list.append({'id': data['id'], 'invalid': True, 'reason': 'api_timeout'})
                return new_data_list, 0, 0.0, 0.0, args, is_invalid_user
            rec_agent_response = rec_agent.act(data)
            if rec_agent_response is None:
                retry_count += 1
                print(f"[WARNING] RecAgent API returned None for data id {data['id']}, retry {retry_count}/{max_retries}")
                if retry_count >= max_retries:
                    print(f"[ERROR] RecAgent failed after {max_retries} retries for data id {data['id']}")
                    is_invalid_user = True
                    # Add minimal data for tracking invalid user
                    if len(new_data_list) == 0:
                        new_data_list.append({'id': data['id'], 'invalid': True, 'reason': 'recagent_failed'})
                    return new_data_list, 0, 0.0, 0.0, args, is_invalid_user
                continue
            print(f"\n[RecAgent Response - Epoch {epoch}, User {data['id']}]")
            print(f"{rec_agent_response}")
            print("-" * 80)
            rec_reason, rec_items = split_rec_reponse(rec_agent_response)
            if rec_items is not None and len(rec_items) > 0:
                # Filter out items that are not in the candidate list
                # Use ID-based matching first, then fallback to name/title matching
                import re
                
                # Get candidate IDs (original news IDs like "N1", "N2")
                candidate_ids = data.get('cans_id', [])
                
                # Debug: print candidate info
                if candidate_ids:
                    print(f"[DEBUG] Candidate IDs: {candidate_ids}")
                    print(f"[DEBUG] Candidate names count: {len(data['cans_name'])}, Candidate IDs count: {len(candidate_ids)}")
                else:
                    print("[DEBUG] No candidate IDs found in data")
                
                if not candidate_ids or len(candidate_ids) != len(data['cans_name']):
                    # Fallback to name-based matching if IDs not available
                    print("[WARNING] cans_id not found or mismatch, using name-based matching")
                    candidates_lower = [can.lower().strip() for can in data['cans_name']]
                    valid_items = []
                    invalid_items = []
                    
                    for item in rec_items:
                        item_lower = item.lower().strip()
                        if item_lower in candidates_lower:
                            # Find the original candidate name
                            idx = candidates_lower.index(item_lower)
                            valid_items.append(data['cans_name'][idx])
                        else:
                            invalid_items.append(item)
                            print(f"[WARNING] Recommended item '{item}' is not in candidate list. Filtering out.")
                else:
                    # ID-based matching: create mappings
                    name_to_id = {}
                    id_to_name = {}
                    for name, news_id in zip(data['cans_name'], candidate_ids):
                        name_to_id[name.lower().strip()] = news_id
                        id_to_name[news_id] = name
                    
                    print(f"[DEBUG] Created name_to_id mapping with {len(name_to_id)} entries")
                    print(f"[DEBUG] Sample candidate IDs: {candidate_ids[:3] if len(candidate_ids) >= 3 else candidate_ids}")
                    
                    # Create set for fast lookup
                    candidate_ids_set = set(candidate_ids)
                    
                    valid_items = []
                    invalid_items = []
                    
                    # Pattern to match news IDs (e.g., "N1", "N2", "N123")
                    id_pattern = re.compile(r'\b([Nn]\d+)\b')
                    
                    for item in rec_items:
                        matched = False
                        item_lower = item.lower().strip()
                        
                        print(f"[DEBUG] Processing item: {item[:100]}...")
                        
                        # 1. Try ID-based matching first (extract ID from LLM response)
                        id_matches = id_pattern.findall(item)
                        if id_matches:
                            print(f"[DEBUG] Found ID matches in item: {id_matches}")
                            # Normalize ID (uppercase)
                            for matched_id in id_matches:
                                normalized_id = matched_id.upper()
                                print(f"[DEBUG] Checking if '{normalized_id}' is in candidate_ids_set: {normalized_id in candidate_ids_set}")
                                if normalized_id in candidate_ids_set:
                                    valid_items.append(id_to_name[normalized_id])
                                    matched = True
                                    print(f"[INFO] Matched by ID: '{normalized_id}' -> '{id_to_name[normalized_id]}'")
                                    break
                            if matched:
                                continue
                        else:
                            print(f"[DEBUG] No ID pattern found in item")
                        
                        # 2. Try exact name match
                        if item_lower in name_to_id:
                            matched_id = name_to_id[item_lower]
                            # Verify this ID is in candidate list
                            if matched_id in candidate_ids_set:
                                valid_items.append(id_to_name[matched_id])
                                matched = True
                                print(f"[INFO] Matched by exact name")
                                continue
                        else:
                            print(f"[DEBUG] Item not found in name_to_id mapping")
                        
                        # 3. Try to extract title and match (for partial matches)
                        title_match = re.search(r'Title:\s*(.+?)(?:$|,|Category:)', item, re.IGNORECASE)
                        if title_match:
                            title = title_match.group(1).strip().lower()
                            # Remove any trailing explanation in parentheses
                            title = re.sub(r'\s*\(.*?\)\s*$', '', title).strip()
                            print(f"[DEBUG] Extracted title: '{title[:50]}...'")
                            # Try to find candidate with matching title
                            for can_name, can_id in zip(data['cans_name'], candidate_ids):
                                can_title_match = re.search(r'Title:\s*(.+?)(?:$|,|Category:)', can_name, re.IGNORECASE)
                                if can_title_match:
                                    can_title = can_title_match.group(1).strip().lower()
                                    if title == can_title:
                                        valid_items.append(can_name)
                                        matched = True
                                        print(f"[INFO] Matched by title: '{title[:50]}...' -> '{can_name[:50]}...'")
                                        break
                        
                        if not matched:
                            invalid_items.append(item)
                            print(f"[WARNING] Recommended item '{item[:100]}...' is not in candidate list. Filtering out.")
                
                # Check if we have enough valid items
                if len(valid_items) < len(data['cans_name']):
                    # Need to fill in missing candidates
                    missing_candidates = []
                    for can in data['cans_name']:
                        can_lower = can.lower().strip()
                        if not any(item.lower().strip() == can_lower for item in valid_items):
                            missing_candidates.append(can)
                    
                    # Add missing candidates at the end (in original order)
                    valid_items.extend(missing_candidates)
                    print(f"[INFO] Added {len(missing_candidates)} missing candidates to complete the ranking.")
                
                # Check if groundtruth (correct_answer) is in the ranking
                correct_answer_lower = data['correct_answer'].lower().strip()
                valid_items_lower = [item.lower().strip() for item in valid_items]
                
                if correct_answer_lower not in valid_items_lower:
                    print(f"[WARNING] Groundtruth '{data['correct_answer']}' is not in the ranking. Retrying...")
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"[ERROR] Groundtruth not found in ranking after {max_retries} retries. Marking user as invalid.")
                        is_invalid_user = True
                        # Add minimal data for tracking invalid user
                        if len(new_data_list) == 0:
                            new_data_list.append({'id': data['id'], 'invalid': True, 'reason': 'groundtruth_not_in_ranking'})
                        return new_data_list, 0, 0.0, 0.0, args, is_invalid_user
                    continue
                
                # Ensure we have exactly the right number of items
                if len(valid_items) == len(data['cans_name']):
                    # Use the top-ranked item as the recommendation
                    rec_item = valid_items[0]
                    rec_ranking = valid_items  # Store filtered ranking
                    if invalid_items:
                        print(f"[INFO] Using filtered ranking (removed {len(invalid_items)} invalid items)")
                    break
                else:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"[ERROR] Could not create valid ranking after {max_retries} retries")
                        is_invalid_user = True
                        # Add minimal data for tracking invalid user
                        if len(new_data_list) == 0:
                            new_data_list.append({'id': data['id'], 'invalid': True, 'reason': 'invalid_ranking'})
                        return new_data_list, 0, 0.0, 0.0, args, is_invalid_user
                    continue
            retry_count += 1
            if retry_count >= max_retries:
                print(f"[ERROR] Failed to parse rec_item after {max_retries} retries for data id {data['id']}")
                is_invalid_user = True
                # Add minimal data for tracking invalid user
                if len(new_data_list) == 0:
                    new_data_list.append({'id': data['id'], 'invalid': True, 'reason': 'parse_failed'})
                return new_data_list, 0, 0.0, 0.0, args, is_invalid_user
        if args.max_epoch == 1:
            new_data = {'id':data['id'],'seq_name':data['seq_name'], 'cans_name':data['cans_name'], 'correct_answer':data['correct_answer'], 'epoch':epoch, 'rec_res':rec_agent_response, 'user_res':None,'prior_answer':data['prior_answer'], 'rec_ranking':rec_ranking}
            new_data_list.append(new_data)
            
            memory_info = {"epoch":epoch, "rec_reason":rec_reason, "rec_item":rec_item, "user_reason":None}
            rec_agent.update_memory(memory_info)
            user_agent.update_memory(memory_info)
            break
        
        #user agent
        retry_count = 0
        max_retries = 10
        api_start_time = time.time()
        max_api_time = 300  # 5 minutes timeout per API call attempt
        while retry_count < max_retries:
            # Check if we've been trying too long
            if time.time() - api_start_time > max_api_time:
                print(f"[ERROR] UserAgent API call timeout after {max_api_time} seconds for data id {data['id']}")
                flag = False  # Assume rejection
                user_reason = "API timeout"
                break
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
            print(f"\n[UserAgent Response - Epoch {epoch}, User {data['id']}]")
            print(f"{user_agent_response}")
            print("-" * 80)
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
        new_data = {'id':data['id'],'seq_name':data['seq_name'], 'cans_name':data['cans_name'], 'correct_answer':data['correct_answer'], 'epoch':epoch, 'rec_res':rec_agent_response, 'user_res':user_agent_response,'prior_answer':data['prior_answer'], 'rec_ranking':rec_ranking}
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
    # Calculate metrics using the ranking
    correct_answer = data['correct_answer'].lower().strip()
    
    # Use the ranking from RecAgent
    rec_ranking_lower = [item.lower().strip() for item in rec_ranking] if rec_ranking else []
    
    # Hit@1: whether the top-ranked item matches the correct answer
    hit_at_1 = 1 if len(rec_ranking_lower) > 0 and rec_ranking_lower[0] == correct_answer else 0
    
    # Calculate MRR (Mean Reciprocal Rank)
    # MRR = 1/rank where rank is the position of the first relevant item in the ranking
    mrr = 0.0
    if correct_answer in rec_ranking_lower:
        # Find the rank (1-indexed) of correct answer in ranking
        rank = rec_ranking_lower.index(correct_answer) + 1
        mrr = 1.0 / rank
    # If correct answer not in ranking, MRR = 0.0
    
    # Calculate nDCG@5 (Normalized Discounted Cumulative Gain at 5)
    # Use the ranking provided by RecAgent (top 5)
    top5_ranked = rec_ranking_lower[:5]
    
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
    
    return new_data_list, hit_at_1, mrr, ndcg5, args, is_invalid_user

def setcallback(x):
    global finish_num
    global total
    global correct
    global mrr_sum
    global ndcg5_sum
    global invalid_users

    if len(x) == 6:
        data_list, hit_at_1, mrr, ndcg5, args, is_invalid_user = x
    else:
        # Backward compatibility
        data_list, hit_at_1, mrr, ndcg5, args = x
        is_invalid_user = False
    
    if is_invalid_user:
        invalid_users += 1
        user_id = "unknown"
        reason = "unknown"
        if len(data_list) > 0:
            if 'id' in data_list[0]:
                user_id = data_list[0]['id']
            if 'reason' in data_list[0]:
                reason = data_list[0]['reason']
        print(f"[INVALID USER] User {user_id} marked as invalid (reason: {reason})")
    else:
        for data in data_list:
            append_jsonl(args.output_file, data)
    
    finish_num += 1
    correct += hit_at_1
    mrr_sum += mrr
    ndcg5_sum += ndcg5
    
    valid_users = finish_num - invalid_users
    print("==============")
    current_hit1 = (correct/valid_users if valid_users > 0 else 0)
    current_mrr = (mrr_sum/valid_users if valid_users > 0 else 0)
    current_ndcg5 = (ndcg5_sum/valid_users if valid_users > 0 else 0)
    global_hit1 = (correct/total if total > 0 else 0)
    global_mrr = (mrr_sum/total if total > 0 else 0)
    global_ndcg5 = (ndcg5_sum/total if total > 0 else 0)
    print(f"current Hit@1 = {correct} / {valid_users} = {current_hit1:.4f}")
    print(f"current MRR = {mrr_sum} / {valid_users} = {current_mrr:.4f}")
    print(f"current nDCG@5 = {ndcg5_sum} / {valid_users} = {current_ndcg5:.4f}")
    print(f"global Hit@1 = {correct} / {total} = {global_hit1:.4f}")
    print(f"global MRR = {mrr_sum} / {total} = {global_mrr:.4f}")
    print(f"global nDCG@5 = {ndcg5_sum} / {total} = {global_ndcg5:.4f}")
    print(f"Invalid users: {invalid_users} / {finish_num}")
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
    
    # Limit the number of users if specified
    if args.max_users is not None and args.max_users > 0:
        original_count = len(merge_data_list)
        merge_data_list = merge_data_list[:args.max_users]
        print(f"Limited evaluation to {len(merge_data_list)} users (from {original_count} total users)")
    
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
    
    # Add progress monitoring to prevent hanging
    import threading
    def monitor_progress():
        last_finish = finish_num
        while True:
            time.sleep(30)  # Check every 30 seconds
            current_finish = finish_num
            if current_finish == last_finish and current_finish < total:
                print(f"[PROGRESS] Still processing... Completed: {current_finish}/{total} ({current_finish/total*100:.1f}%)")
            last_finish = current_finish
            if current_finish >= total:
                break
    
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    
    try:
        pool.join()
        print("All tasks completed!")
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted by user. Terminating pool...")
        pool.terminate()
        pool.join()
        print("[INFO] Pool terminated.")
    
    # Clear model cache after all processes finish
    # Note: Each process has its own memory space, so when a process terminates,
    # its memory (including the cached model) is automatically freed by the OS.
    # This is just for clarity and to help with garbage collection in the main process.
    global _process_model_cache
    _process_model_cache = None
    
    # Print final metrics
    valid_users = total - invalid_users
    final_hit1 = (correct/valid_users if valid_users > 0 else 0)
    final_mrr = (mrr_sum/valid_users if valid_users > 0 else 0)
    final_ndcg5 = (ndcg5_sum/valid_users if valid_users > 0 else 0)
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {total}")
    print(f"Valid users: {valid_users}")
    print(f"Invalid users: {invalid_users}")
    print(f"Hit@1: {correct} / {valid_users} = {final_hit1:.4f}")
    print(f"MRR: {mrr_sum} / {valid_users} = {final_mrr:.4f}")
    print(f"nDCG@5: {ndcg5_sum} / {valid_users} = {final_ndcg5:.4f}")
    print("="*50)

if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    main(args)
    