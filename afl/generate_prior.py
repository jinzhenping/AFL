"""
Generate prior recommendations using SASRec model for MIND dataset.
This script creates a CSV file with format: id,generate,real
"""
import os
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.mind_dataset import MindDataset
from utils.model import SASRec


def get_args():
    parser = argparse.ArgumentParser(description='Generate prior recommendations using SASRec model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to MIND dataset directory')
    parser.add_argument('--model_path', type=str, default=None, help='Path to SASRec model. Defaults to ./SASRec.pth if not specified.')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file path (e.g., ./data/prior_mind/SASRec.csv)')
    parser.add_argument('--stage', type=str, default='test', help='Dataset stage: test')
    parser.add_argument('--cans_num', type=int, default=5, help='Number of candidates')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID to use')
    return parser.parse_args()


def generate_prior_recommendations(args):
    """Generate prior recommendations for MIND dataset"""
    
    # Load dataset
    print("Loading dataset...")
    dataset = MindDataset(args.data_dir, args.stage, args.cans_num, ", ", True)
    print(f"Dataset size: {len(dataset)}")
    
    # Get device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None and torch.cuda.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and get item mappings from dataset
    print("Loading model...")
    
    # Get item_num from dataset
    item_num = len(dataset.item_id2name)
    seq_size = 50  # Default sequence size for MIND
    
    # Load model
    # Try to load as full model first (common case)
    loaded = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Check if it's a full model object (has forward method)
    if hasattr(loaded, 'forward') or hasattr(loaded, 'forward_eval'):
        # It's a full model object
        model = loaded
        model.to(device)
        print("Loaded as full model object")
    elif isinstance(loaded, dict):
        # It's a state_dict, try to load into new model
        model = SASRec(64, item_num, seq_size, 0.1, device)
        model.to(device)
        
        if 'state_dict' in loaded:
            state_dict = loaded['state_dict']
        else:
            state_dict = loaded
        
        # Try loading with strict=False first (in case of minor mismatches)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            print("Warning: Strict loading failed, trying with strict=False...")
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Loaded with strict=False (some keys may be missing)")
            except Exception as e:
                print(f"Error loading state_dict: {e}")
                raise
    else:
        raise ValueError(f"Unknown model format loaded from {args.model_path}")
    
    model.eval()
    print("Model loaded successfully")
    
    # Get id2name mapping from dataset
    id2name = dataset.item_id2name  # Maps string ID (e.g., "N1") to formatted name
    int2news_id = dataset.int2news_id  # Maps integer ID to string ID (e.g., "N1")
    
    # Generate recommendations
    print("Generating recommendations...")
    results = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            
            # Get the sequence and candidates
            seq = data['seq']
            len_seq = data['len_seq']
            candidates = data['cans']  # Integer IDs
            correct_answer = data['correct_answer']  # Title
            
            # Generate prediction using SASRec model
            seq_b = [seq]
            # Ensure len_seq doesn't exceed the padded sequence length
            actual_seq_len = min(len_seq, len(seq))
            len_seq_b = [actual_seq_len]
            states = np.array(seq_b)
            states = torch.LongTensor(states)
            states = states.to(device)
            
            prediction = model.forward_eval(states, np.array(len_seq_b))
            
            # Mask out items not in candidates
            sampling_idx = [True] * item_num
            for can_id in candidates:
                if can_id < item_num:
                    sampling_idx[can_id] = False
            
            sampling_idxs = [torch.tensor(sampling_idx)]
            sampling_idxs = torch.stack(sampling_idxs, dim=0)
            prediction = prediction.cpu().detach().masked_fill(sampling_idxs, prediction.min().item() - 1)
            
            # Get top-1 recommendation
            values, topK = prediction.topk(1, dim=1, largest=True, sorted=True)
            recommended_id_int = int(topK.numpy()[0][0])  # Integer ID from model
            
            # Helper function to convert integer ID to name
            def int_id_to_name(int_id):
                """Convert integer ID to formatted name string"""
                if int_id in int2news_id:
                    news_id_str = int2news_id[int_id]
                    if news_id_str in id2name:
                        return id2name[news_id_str]
                return None
            
            # Convert integer ID to string ID, then to name
            recommended_name = int_id_to_name(recommended_id_int)
            
            # Fallback: use first candidate name if recommended_id not found
            if recommended_name is None:
                if candidates and len(candidates) > 0:
                    recommended_name = int_id_to_name(candidates[0])
                    if recommended_name is None:
                        recommended_name = "Unknown"
                else:
                    recommended_name = "Unknown"
            
            results.append({
                'id': i,
                'generate': recommended_name,
                'real': correct_answer
            })
    
    # Calculate accuracy metrics
    print("\n" + "="*80)
    print("PRIOR RECOMMENDATION ACCURACY METRICS")
    print("="*80)
    
    total = len(results)
    correct = 0
    
    # Hit@1: Exact match between generate and real
    for result in results:
        generate = result['generate'].lower().strip()
        real = result['real'].lower().strip()
        if generate == real:
            correct += 1
    
    hit_at_1 = correct / total if total > 0 else 0.0
    
    print(f"Total samples: {total}")
    print(f"Hit@1 (Exact Match): {correct} / {total} = {hit_at_1:.4f} ({hit_at_1*100:.2f}%)")
    
    # Additional: Check for partial matches (if titles contain similar keywords)
    # This is optional and can be useful for understanding near-misses
    partial_matches = 0
    for result in results:
        generate = result['generate'].lower().strip()
        real = result['real'].lower().strip()
        if generate != real:
            # Check if they share significant words (more than 50% of words match)
            generate_words = set(generate.split())
            real_words = set(real.split())
            if len(generate_words) > 0 and len(real_words) > 0:
                common_words = generate_words.intersection(real_words)
                similarity = len(common_words) / max(len(generate_words), len(real_words))
                if similarity > 0.5:  # More than 50% word overlap
                    partial_matches += 1
    
    print(f"Partial matches (>50% word overlap): {partial_matches} / {total - correct} (excluding exact matches)")
    print("="*80 + "\n")
    
    # Save to CSV
    print(f"Saving results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"Saved {len(results)} recommendations to {args.output_file}")
    
    return results


if __name__ == '__main__':
    args = get_args()
    
    # Set default model path if not specified
    if args.model_path is None:
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_model_path = os.path.join(current_dir, 'SASRec.pth')
        if os.path.exists(default_model_path):
            args.model_path = default_model_path
            print(f"Using default model path: {args.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found. Please specify --model_path or place SASRec.pth in project root: {default_model_path}")
    
    generate_prior_recommendations(args)

