import os
import time
import argparse
import json
import jsonlines
import torch
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import sys
import pandas as pd
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.regular_function import split_user_response, split_rec_reponse
from utils.rw_process import append_jsonl, write_jsonl, read_jsonl
from utils.api_request import api_request
from utils.model import SASRec

class RecAgent:
    def __init__(self, args, mode='prior_rec'):
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        self.load_prompt()

    def load_prompt(self):
        if self.mode =='prior_rec':
            if 'lastfm' in self.args.data_dir:
                from constant.lastfm_prior_model_prompt import rec_system_prompt, rec_user_prompt, rec_memory_system_prompt, rec_memory_user_prompt, rec_build_memory
            elif 'mind' in self.args.data_dir.lower():
                from constant.mind_prior_model_prompt import rec_system_prompt, rec_user_prompt, rec_memory_system_prompt, rec_memory_user_prompt, rec_build_memory
            else:
                raise ValueError("Invalid mode: {}".format(self.args.data_dir))
            self.rec_system_prompt = rec_system_prompt
            self.rec_user_prompt = rec_user_prompt
            self.rec_memory_system_prompt = rec_memory_system_prompt
            self.rec_memory_user_prompt = rec_memory_user_prompt
            self.rec_build_memory = rec_build_memory
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def act(self, data, reason=None, item=None):
        if self.mode =='prior_rec':
            # Ensure prior_answer is a string
            prior_answer = data.get('prior_answer', '')
            if isinstance(prior_answer, dict):
                prior_answer = str(prior_answer)
            elif not isinstance(prior_answer, str):
                prior_answer = str(prior_answer) if prior_answer is not None else ''
            
            if len(self.memory) == 0:
                system_prompt = self.rec_system_prompt
                user_prompt = self.rec_user_prompt.format(data['seq_str'], data['len_cans'],data['cans_str'], prior_answer)
            else:
                system_prompt = self.rec_memory_system_prompt
                user_prompt = self.rec_memory_user_prompt.format(data['seq_str'],data['len_cans'], data['cans_str'], '\n'.join(self.memory))
            response = api_request(system_prompt, user_prompt, self.args)
            return response
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))
        
    def build_memory(self, info):
        return self.rec_build_memory.format(info['epoch'], info['rec_item'], info['rec_reason'], info['user_reason'])
    
    def update_memory(self, info):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        write_jsonl(path, self.info_list)
    
    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]


class UserModelAgent:
    def __init__(self, args, mode='prior_rec'):
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        # Set device based on gpu argument or default
        if args.gpu is not None:
            if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
                self.device = torch.device(f"cuda:{args.gpu}")
            else:
                if not torch.cuda.is_available():
                    print(f"Warning: CUDA not available, using CPU instead of GPU {args.gpu}")
                else:
                    print(f"Warning: GPU {args.gpu} not available (only {torch.cuda.device_count()} GPUs), using default CUDA device")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_prompt()
        self.load_model()
        self.id2name = dict()
        self.name2id = dict()
        self.build_id2name()

    def build_id2name(self):
        if 'movielens' in self.args.data_dir:
            def get_mv_title(s):
                sub_list=[", The", ", A", ", An"]
                for sub_s in sub_list:
                    if sub_s in s:
                        return sub_s[2:]+" "+s.replace(sub_s,"")
                return s
            item_path = os.path.join(self.args.data_dir, 'u.item')
            with open(item_path, 'r', encoding = "ISO-8859-1") as f:
                for line in f.readlines():
                    features = line.strip('\n').split('|')
                    id = int(features[0]) - 1
                    name = get_mv_title(features[1][:-7])
                    self.id2name[id] = name
                    self.name2id[name] = id
        elif 'lastfm' in self.args.data_dir or 'steam' in self.args.data_dir:
            if '_ab' in self.args.data_dir:
                item_path=os.path.join(self.args.data_dir, 'id2name_AB_llara.txt')
            else:
                item_path=os.path.join(self.args.data_dir, 'id2name.txt')
            with open(item_path, 'r') as f:
                for l in f.readlines():
                    ll = l.strip('\n').split('::')
                    self.id2name[int(ll[0])] = ll[1].strip()
                    self.name2id[ll[1].strip()] = int(ll[0])
        elif 'mind' in self.args.data_dir.lower():
            # Load news ID to name mapping from MIND_news.tsv
            news_file = os.path.join(self.args.data_dir, 'MIND_news.tsv')
            if not os.path.exists(news_file):
                raise FileNotFoundError(f"MIND_news.tsv not found at {news_file}")
            
            # Create a mapping from news ID (string) to integer ID for model compatibility
            # Also create name mapping
            with open(news_file, 'r', encoding='utf-8') as f:
                news_id_to_int = {}
                int_id = 0
                for line in f.readlines():
                    parts = line.strip('\n').split('\t')
                    if len(parts) >= 4:
                        news_id = parts[0]  # e.g., "N1"
                        title = parts[3]  # Title is the 4th column
                        news_id_to_int[news_id] = int_id
                        self.id2name[int_id] = title.strip()
                        self.name2id[title.strip()] = int_id
                        int_id += 1
            # Store the mapping for later use
            self.news_id_to_int = news_id_to_int
        else:
            raise ValueError("Invalid data dir: {}".format(self.args.data_dir))
        
    def load_model(self):
        print("loading model")
        data_directory = self.args.data_dir
        
        # Handle MIND dataset which may not have data_statis.df
        if 'mind' in data_directory.lower():
            # For MIND dataset, calculate parameters from data
            news_file = os.path.join(data_directory, 'MIND_news.tsv')
            if os.path.exists(news_file):
                # Count number of news items
                with open(news_file, 'r', encoding='utf-8') as f:
                    item_num = sum(1 for line in f if line.strip())
                self.item_num = item_num
            else:
                # Fallback: use id2name mapping length
                self.item_num = len(self.id2name)
            
            # Default sequence size for MIND (can be adjusted)
            self.seq_size = 50
        else:
            # For other datasets, use data_statis.df
            data_statis_path = os.path.join(data_directory, 'data_statis.df')
            if os.path.exists(data_statis_path):
                data_statis = pd.read_pickle(data_statis_path)
                self.seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
                self.item_num = data_statis['item_num'][0]  # total number of items
            else:
                raise FileNotFoundError(f"data_statis.df not found at {data_statis_path}")
        
        # Determine model path - use SASRec.pth in root if not specified
        if self.args.model_path is None:
            # Try to find SASRec.pth in project root
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_model_path = os.path.join(current_dir, 'SASRec.pth')
            if os.path.exists(default_model_path):
                model_path = default_model_path
                print(f"Using default model path: {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found. Please specify --model_path or place SASRec.pth in project root: {default_model_path}")
        else:
            model_path = self.args.model_path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at specified path: {model_path}")
        
        # Try to load as full model first (common case)
        loaded = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Check if it's a full model object (has forward method)
        if hasattr(loaded, 'forward') or hasattr(loaded, 'forward_eval'):
            # It's a full model object
            self.model = loaded
            self.model.to(self.device)
            print("Loaded as full model object")
        elif isinstance(loaded, dict):
            # It's a state_dict, try to load into new model
            self.model = SASRec(64, self.item_num, self.seq_size, 0.1, self.device)
            self.model.to(self.device)
            
            if 'state_dict' in loaded:
                state_dict = loaded['state_dict']
            else:
                state_dict = loaded
            
            # Try loading with strict=False first (in case of minor mismatches)
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                # Only show warning once, not for every process
                if not hasattr(self, '_load_warning_shown'):
                    print("Warning: Model structure mismatch, loading with strict=False (this is usually fine)")
                    self._load_warning_shown = True
                try:
                    self.model.load_state_dict(state_dict, strict=False)
                except Exception as e2:
                    print(f"Error loading state_dict: {e2}")
                    raise
        else:
            raise ValueError(f"Unknown model format loaded from {model_path}")
        
        print("load model success")
    
    def load_prompt(self):
        if self.mode == 'prior_rec':
            if 'lastfm' in self.args.data_dir:
                from constant.lastfm_prior_model_prompt import user_system_promt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_recommend_system_prompt, user_recommend_user_prompt, user_recommend_memory_system_prompt, user_recommend_memory_user_prompt, user_build_memory, user_build_memory_2
            elif 'mind' in self.args.data_dir.lower():
                from constant.mind_prior_model_prompt import user_system_promt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
                # MIND uses same prompts for memory and non-memory cases
                self.user_recommend_system_prompt = user_system_promt
                self.user_recommend_user_prompt = user_user_prompt
                self.user_recommend_memory_system_prompt = user_memory_system_prompt
                self.user_recommend_memory_user_prompt = user_memory_user_prompt
            else:
                raise ValueError("Invalid dataset: {}".format(self.args.data_dir))
            self.user_system_promt = user_system_promt
            self.user_user_prompt = user_user_prompt
            self.user_memory_system_prompt = user_memory_system_prompt
            self.user_memory_user_prompt = user_memory_user_prompt
            if 'mind' not in self.args.data_dir.lower():
                self.user_recommend_system_prompt = user_recommend_system_prompt
                self.user_recommend_user_prompt = user_recommend_user_prompt
                self.user_recommend_memory_system_prompt = user_recommend_memory_system_prompt
                self.user_recommend_memory_user_prompt = user_recommend_memory_user_prompt
            self.user_build_memory = user_build_memory
            self.user_build_memory_2 = user_build_memory_2

        elif self.mode == 'pred':
            if 'lastfm' in self.args.data_dir:
                from constant.lastfm_ab_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2, user_memory_system_prompt, user_memory_user_prompt
            elif 'mind' in self.args.data_dir.lower():
                from constant.mind_prior_model_prompt import user_system_promt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
                # For pred mode, use the same prompts
                user_system_prompt = user_system_promt
            else:
                raise ValueError("Invalid dataset: {}".format(self.args.data_dir))
            self.user_system_prompt = user_system_prompt
            self.user_user_prompt = user_user_prompt
            self.user_memory_system_prompt = user_memory_system_prompt
            self.user_memory_user_prompt = user_memory_user_prompt
            self.user_build_memory = user_build_memory
            self.user_build_memory_2 = user_build_memory_2
    def act(self, data, reason=None, item=None):
        if self.mode == 'prior_rec':
            model_output = self.model_generate(data['seq'], data['len_seq'], data['cans'])
            
            # Ensure prior_answer is a string
            prior_answer = data.get('prior_answer', '')
            if isinstance(prior_answer, dict):
                prior_answer = str(prior_answer)
            elif not isinstance(prior_answer, str):
                prior_answer = str(prior_answer) if prior_answer is not None else ''
            
            if len(self.memory) == 0:
                system_prompt = self.user_system_promt.format(data['seq_str'], prior_answer)
            else:
                system_prompt = self.user_system_promt.format(data['seq_str'], prior_answer)
            user_prompt = self.user_user_prompt.format(data['cans_str'],model_output, item, reason)
            response = api_request(system_prompt, user_prompt, self.args)
            return response
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def pred_model(self, data, score):
        if len(self.memory) == 0:
            system_prompt = self.user_system_prompt.format(data['seq_str'])
            user_prompt = self.user_user_prompt.format(data['pred_item'], score)
        else:
            system_prompt = self.user_memory_system_prompt.format(data['seq_str'], data['cans_str'],'\n'.join(self.memory))
            user_prompt = self.user_memory_user_prompt.format(data['pred_item'],score)
            
        response = api_request(system_prompt, user_prompt, self.args)
        return response
    
    def build_memory(self, info):
        if info['user_reason'] is not None:
            return self.user_build_memory.format(info['epoch'], info['rec_item'], info['rec_reason'], info['user_reason'])
        else:
            return self.user_build_memory_2.format(info['epoch'], info['rec_item'], info['rec_reason'])
    
    def update_memory(self, info):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        write_jsonl(path, self.info_list)
    
    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]

    def model_generate(self, seq, len_seq, candidates):
        seq_b = [seq]
        len_seq_b = [len_seq]
        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(self.device)
        prediction = self.model.forward_eval(states, np.array(len_seq_b))

        sampling_idx=[True]*self.item_num
        cans_num = len(candidates)
        for i in candidates:
            sampling_idx.__setitem__(i,False)
        sampling_idxs = [torch.tensor(sampling_idx)]
        sampling_idxs=torch.stack(sampling_idxs,dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs,prediction.min().item()-1)
        values, topK = prediction.topk(cans_num, dim=1, largest=True, sorted=True)
        topK = topK.numpy()[0]
        name_list = [self.id2name[id] for id in topK]
        len_ret = int(len(name_list) /4 )
        return ', '.join(name_list[:len_ret])

    def score(self, seq, len_seq, candidates):
        #print("seq = ", seq)
        #print("len seq = ", len_seq)
        #print("cans = ", candidates)
        seq_b = [seq]
        len_seq_b = [len_seq]
        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(self.device)
        # pred
        prediction = self.model.forward_eval(states, np.array(len_seq_b))

        sampling_idx=[True]*self.item_num
        cans_num = len(candidates)
        for i in candidates:
            sampling_idx.__setitem__(i,False)
        sampling_idxs = [torch.tensor(sampling_idx)]
        sampling_idxs=torch.stack(sampling_idxs,dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs,prediction.min().item()-1)
        values, topK = prediction.topk(cans_num, dim=1, largest=True, sorted=True)
        values = values.numpy()[0]
        topK = topK.numpy()[0]
        score_dict = {}
        for i in range(len(topK)):
            id = topK[i]
            score = values[i]
            name = self.id2name[id]
            score_dict[name] = score
        return score_dict