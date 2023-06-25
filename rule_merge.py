import os
import pickle
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from constant import ROOT_DIR


'''
if not os.path.exists(Path(ROOT_DIR) / 'dataset\processed\log_cnt_dict.pkl'):
    log_df = pd.read_json(Path(ROOT_DIR) / "dataset/raw/item_share_train_info.json")
    log_cnt = log_df.groupby(['inviter_id', 'voter_id'])['timestamp'].count().reset_index()
    log_cnt = log_cnt.sort_values(['inviter_id', 'timestamp'], ascending=False).reset_index(drop=True)
    log_cnt_dict = log_cnt.groupby('inviter_id').agg(list)[['voter_id']].to_dict()['voter_id']

    with open(Path(ROOT_DIR) / 'dataset\processed\log_cnt_dict.pkl', 'wb') as f:
        pickle.dump(log_cnt_dict, f)
else:
    with open(Path(ROOT_DIR) / 'dataset\processed\log_cnt_dict.pkl', 'rb') as f:
        log_cnt_dict = pickle.load(f)
'''

def run_merge():
    rank_avg_list = list()
    submit_list = []
    for model_id in range(20):
        with open(Path(ROOT_DIR) / f'dataset/submit/submit_DisMult_GES_{model_id}.json', 'r') as f:
            submit = json.load(f)
        
        submit_list.append(submit)

    #w = [1, 1.2, 1.5, 1.8, 2] #0.3043
    w = [1, 1, 1, 1, 1] #0.3046
    #w = [1, 1, 1, 2, 2] #0.3043
    for rank_dct_tuple in tqdm(zip(*submit_list)):
        merge_dict = dict()
        for i in range(20):
            rank_dct = rank_dct_tuple[i]
            tid = rank_dct["triple_id"]
            voters = rank_dct["candidate_voter_list"]
            
            for rank, candi in enumerate(voters):
                merge_dict.setdefault(candi, 0)
                merge_dict[candi] += w[i%5] * (1 / (1 + rank))
    
        ret_vals = [candi[0] for candi in sorted(merge_dict.items(), key=lambda x: x[1], reverse=True)]
    
        rank_avg_list.append(
            {'triple_id': tid,
             'candidate_voter_list': ret_vals[: 5]}
        )

    with open(Path(ROOT_DIR) / 'dataset/submit/submit_ensemble.json', 'w') as f:
        json.dump(rank_avg_list, f)

if __name__=='__main__':
    run_merge()
        