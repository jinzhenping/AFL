import torch
import os
import torch.utils.data as data
import pandas as pd
import random

class MindDataset(data.Dataset):
    def __init__(self, data_dir=r'./mind', stage='test', cans_num=5, sep=", ", no_augment=True):
        self.data_dir = data_dir
        self.cans_num = cans_num
        self.stage = stage
        self.sep = sep
        self.aug = (stage=='train') and not no_augment
        self.padding_item_id = 0  # MIND uses 0 as padding or we can use a special token
        self.check_files()
    
    def __len__(self):
        return len(self.session_data)
    
    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        
        # Parse click history (second column) - string IDs
        click_history = temp['click_history']
        seq_unpad_str = click_history if isinstance(click_history, list) else click_history.split()
        
        # Parse candidates (third column) - first one is ground truth - string IDs
        candidates_raw = temp['candidates']
        candidates_list_str = candidates_raw if isinstance(candidates_raw, list) else candidates_raw.split()
        
        # Ground truth is the first candidate (string ID)
        next_item_str = candidates_list_str[0]
        
        # Convert string IDs to integer IDs for model
        seq_unpad_int = [self.news_id2int.get(item, 0) for item in seq_unpad_str if item in self.news_id2int]
        candidates_list_int = [self.news_id2int.get(can, 0) for can in candidates_list_str if can in self.news_id2int]
        next_item_int = self.news_id2int.get(next_item_str, 0)
        
        # Use provided candidates or sample if needed
        if len(candidates_list_int) >= self.cans_num:
            candidates_int = candidates_list_int[:self.cans_num]
            candidates_str = candidates_list_str[:self.cans_num]
        else:
            # If not enough candidates, use all and pad with negative samples
            candidates_int = candidates_list_int.copy()
            candidates_str = candidates_list_str.copy()
            all_int_ids = set(self.news_id2int.values())
            negative_candidates_int = list(all_int_ids - set(seq_unpad_int) - set(candidates_list_int))
            negative_candidates_int_sample = random.sample(negative_candidates_int, self.cans_num - len(candidates_int))
            candidates_int.extend(negative_candidates_int_sample)
            # Convert back to string IDs for negative samples
            for neg_int in negative_candidates_int_sample:
                neg_str = self.int2news_id.get(neg_int, '')
                if neg_str:
                    candidates_str.append(neg_str)
            random.shuffle(candidates_int)
            # Shuffle string candidates in same order
            candidates_str = [self.int2news_id.get(cid, '') for cid in candidates_int if cid in self.int2news_id]
        
        # Convert to names for display
        seq_title = [self.item_id2name.get(item, item) for item in seq_unpad_str if item in self.item_id2name]
        cans_name = [self.item_id2name.get(can, can) for can in candidates_str if can in self.item_id2name]
        next_item_name = self.item_id2name.get(next_item_str, next_item_str)
        
        # Format candidates with ID for LLM prompt (e.g., "N1: Category: ..., Subcategory: ..., Title: ...")
        cans_str_with_id = []
        for can_id, can_name in zip(candidates_str, cans_name):
            cans_str_with_id.append(f"{can_id}: {can_name}")
        
        # Pad sequence for model input (use integer IDs)
        seq = self.pad_sequence(seq_unpad_int)
        len_seq = len(seq_unpad_int)
        
        sample = {
            'id': i,
            'seq': seq,  # Integer IDs for model
            'seq_name': seq_title,  # Titles for display
            'len_seq': len_seq,
            'seq_str': self.sep.join(seq_title),
            'cans': candidates_int,  # Integer IDs for model
            'cans_name': cans_name,  # Titles for display (without ID)
            'cans_id': candidates_str,  # Original news IDs (e.g., "N1", "N2")
            'cans_str': self.sep.join(cans_str_with_id),  # Formatted with ID for LLM prompt
            'len_cans': len(candidates_int),
            'item_id': next_item_int,  # Integer ID for model
            'item_name': next_item_name,  # Title for display
            'correct_answer': next_item_name,  # Title for evaluation
            'correct_answer_id': next_item_str  # Original news ID for evaluation
        }
        return sample
    
    def pad_sequence(self, seq, max_len=50):
        """Pad sequence to max_len"""
        padded = seq[:max_len] if len(seq) > max_len else seq
        while len(padded) < max_len:
            padded.append(self.padding_item_id)
        return padded
    
    def check_files(self):
        # Load news metadata
        self.item_id2name = self.get_news_id2name()
        
        # Load user interaction data
        mind_file = os.path.join(self.data_dir, 'MIND.tsv')
        if not os.path.exists(mind_file):
            raise FileNotFoundError(f"MIND.tsv not found at {mind_file}")
        
        self.session_data = self.load_mind_data(mind_file)
    
    def get_news_id2name(self):
        """Load news ID to formatted info (category, subcategory, title) from MIND_news.tsv and create string to int mapping"""
        news_id2name = dict()
        news_id2int = dict()
        news_id2info = dict()  # Store full info (category, subcategory, title)
        news_file = os.path.join(self.data_dir, 'MIND_news.tsv')
        
        if not os.path.exists(news_file):
            raise FileNotFoundError(f"MIND_news.tsv not found at {news_file}")
        
        int_id = 0
        with open(news_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip('\n').split('\t')
                if len(parts) >= 4:
                    news_id = parts[0]  # e.g., "N1"
                    category = parts[1] if len(parts) > 1 else ""
                    subcategory = parts[2] if len(parts) > 2 else ""
                    title = parts[3] if len(parts) > 3 else ""
                    
                    # Format: "Category: {category}, Subcategory: {subcategory}, Title: {title}"
                    formatted_info = f"Category: {category}, Subcategory: {subcategory}, Title: {title.strip()}"
                    news_id2name[news_id] = formatted_info
                    news_id2info[news_id] = {
                        'category': category,
                        'subcategory': subcategory,
                        'title': title.strip()
                    }
                    news_id2int[news_id] = int_id
                    int_id += 1
        
        # Store the mapping for converting string IDs to integers
        self.news_id2int = news_id2int
        self.int2news_id = {v: k for k, v in news_id2int.items()}
        self.news_id2info = news_id2info  # Store full info for potential future use
        
        return news_id2name
    
    def load_mind_data(self, filepath):
        """Load MIND.tsv data"""
        data_list = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip('\n').split('\t')
                if len(parts) >= 3:
                    user_id = parts[0]
                    click_history = parts[1]  # Space-separated news IDs
                    candidates = parts[2]  # Space-separated candidate news IDs (first is ground truth)
                    
                    data_list.append({
                        'user_id': user_id,
                        'click_history': click_history,
                        'candidates': candidates
                    })
        
        df = pd.DataFrame(data_list)
        
        # Filter out users with too short history (at least 3 clicks)
        # Create a boolean mask
        mask = df['click_history'].apply(lambda x: len(x.split()) >= 3 if isinstance(x, str) else len(x) >= 3)
        df = df[mask].reset_index(drop=True)
        
        return df

if __name__ == '__main__':
    dataset = MindDataset(data_dir='./mind', stage='test', cans_num=5)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Sequence: {sample['seq_str'][:100]}...")
    print(f"Candidates: {sample['cans_str'][:100]}...")
    print(f"Correct answer: {sample['correct_answer']}")

