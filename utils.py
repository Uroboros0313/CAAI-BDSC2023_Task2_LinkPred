import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader

from constant import BATCH_SIZE



class KGDataCenter:
    def __init__(self,
                 root,
                 negative_num=1) -> None:
        self.root = Path(root)
        self.negative_num = negative_num
        
        self.itemid2num = dict()
        self.userid2num = dict()
        self.itemnum2id = dict()
        self.usernum2id = dict()
        self.num_user = 0
        self.num_item = 0
        self.user_attrs_enc_map = dict()
        self.item_attrs_enc_map = dict()
        self.user_attrs_dict = dict()
        self.item_attrs_dict = dict()
        self.triplets = None
        self.in_train_rels = None
        
        self.process()
        
    def process(self):
        item_info_df = pd.read_json(self.raw_paths[0])
        user_info_df = pd.read_json(self.raw_paths[1])
        
        self.itemid2num = dict(zip(item_info_df['item_id'].unique(), range(item_info_df['item_id'].nunique()))
                              )
        self.userid2num = dict(zip(user_info_df['user_id'].unique(), range(user_info_df['user_id'].nunique())))
        
        self.itemnum2id = {val: key for key, val in self.itemid2num.items()}
        self.usernum2id = {val: key for key, val in self.userid2num.items()}
        
        self.num_user = len(self.userid2num)
        self.num_item = len(self.itemid2num)
        
        self.user_attrs_enc_map = dict()
        self.item_attrs_enc_map = dict()
        for user_f in self.user_attrs:
            self.user_attrs_enc_map[user_f] = dict(zip(user_info_df[user_f].unique(), 
                                                        range(user_info_df[user_f].nunique())))
            
        for item_f in self.item_attrs:
            self.item_attrs_enc_map[item_f] = dict(zip(item_info_df[item_f].unique(), 
                                                        range(item_info_df[item_f].nunique())))
        
        self.user_attrs_dict = dict()
        self.item_attrs_dict = dict()
        processed_user_info_df = user_info_df.copy()
        for user_attr in self.user_attrs:
            processed_user_info_df[user_attr] =\
                processed_user_info_df[user_attr].map(self.user_attrs_enc_map[user_attr])
                
        processed_item_info_df = item_info_df.copy()
        for item_attr in self.item_attrs:
            processed_item_info_df[item_attr] =\
                processed_item_info_df[item_attr].map(self.item_attrs_enc_map[item_attr])
    
        for user_row in processed_user_info_df.itertuples():
            user_attr_list = list(user_row)[1: ]
            self.user_attrs_dict[self.userid2num[user_attr_list[0]]] = user_attr_list[1: ]
            
        for item_row in processed_item_info_df.itertuples():
            item_attr_list = list(item_row)[1: ]
            self.item_attrs_dict[self.itemid2num[item_attr_list[0]]] = item_attr_list[1: ]
        
        item_share_info_df = pd.read_json(self.raw_paths[2])
        item_share_info_df = item_share_info_df.merge(right=item_info_df,
                                                      how='left',
                                                      on='item_id')
        
        self.triplets = np.vstack([item_share_info_df['inviter_id'].map(self.userid2num).values,
                                   item_share_info_df['item_id'].map(self.itemid2num).values,
                                   item_share_info_df['voter_id'].map(self.userid2num).values]).transpose(1, 0)
        
        num_train = int(self.triplets.shape[0] * 0.8)
        self.train_triplets = self.triplets[: num_train, :]
        self.valid_triplets = self.triplets[num_train :, :]
        
        item_share_test_df = pd.read_json(self.raw_paths[3])
        self.test_tuplets = np.vstack([item_share_test_df['inviter_id'].map(self.userid2num).values,
                                       item_share_test_df['item_id'].map(self.itemid2num).values]).transpose(1, 0)

    def generate_train_dataloader(self, is_dev=True):
        if is_dev:
            train_triplets = self.train_triplets.copy()
        else:
            train_triplets = self.triplets.copy()
        
        self.in_train_rels = set(list(train_triplets[:, 1]))
        
        neg_train_triplets = self.generate_negative_triplets(train_triplets)
        
        train_attrs = self.generate_triplets_attrs(train_triplets)
        neg_train_attrs = self.generate_triplets_attrs(neg_train_triplets)
        
        train_loader = DataLoader(KGTripletDataset(train_triplets, *train_attrs), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True)
        neg_train_loader = DataLoader(KGTripletDataset(neg_train_triplets,*neg_train_attrs), 
                                      batch_size=BATCH_SIZE, 
                                      shuffle=True)
        
        return train_loader, neg_train_loader
    
    def generate_test_dataset(self):
        test_tuplets = self.test_tuplets.copy()
        
        test_attrs = self.generate_triplets_attrs(test_tuplets)
        test_tuplets = th.tensor(test_tuplets, dtype=th.long)
        test_ha = th.tensor(test_attrs[0], dtype=th.long)
        test_ra = th.tensor(test_attrs[1], dtype=th.long)
        
        return (test_tuplets, test_ha, test_ra)
    
    def generate_valid_dataset(self):
        valid_triplets = self.valid_triplets.copy()
        valid_attrs = self.generate_triplets_attrs(valid_triplets)
        
        valid_triplets = th.tensor(valid_triplets, dtype=th.long)
        valid_ha = th.tensor(valid_attrs[0], dtype=th.long)
        valid_ra = th.tensor(valid_attrs[1], dtype=th.long)
        valid_ta = th.tensor(valid_attrs[2], dtype=th.long)
        
        return (valid_triplets, valid_ha, valid_ra, valid_ta)
        
    def generate_negative_triplets(self, orig_triplets, threshold=0.7):
        heads = orig_triplets[:, 0]
        rels = orig_triplets[:, 1]
        tails = orig_triplets[:, 2]
        
        mask_head_prop =\
            np.random.randint(0, 10, orig_triplets.shape[0]) / 10
        neg_users =\
            np.random.randint(0, self.num_user, orig_triplets.shape[0])
        
        neg_heads = np.where(mask_head_prop < threshold, neg_users, heads)
        neg_tails = np.where(mask_head_prop >= threshold, neg_users, tails)
        neg_triplets = np.vstack([neg_heads, rels, neg_tails]).transpose(1, 0)
        
        return neg_triplets
    
    def generate_triplets_attrs(self, orig_triplets):
        heads = orig_triplets[:, 0]
        head_attrs = []
        for idx in range(orig_triplets.shape[0]):
            head_attrs.append(self.user_attrs_dict[heads[idx]])
        
        rels = orig_triplets[:, 1]
        rel_attrs = []
        for idx in range(orig_triplets.shape[0]):
            rel_attrs.append(self.item_attrs_dict[rels[idx]])
            
        if orig_triplets.shape[1] > 2:
            tails = orig_triplets[:, 2]
            tail_attrs = []
            for idx in range(orig_triplets.shape[0]):
                tail_attrs.append(self.user_attrs_dict[tails[idx]])
    
            return np.array(head_attrs), np.array(rel_attrs), np.array(tail_attrs)
        else:
            return np.array(head_attrs), np.array(rel_attrs)
    
                
    @property
    def raw_paths(self):
        _raw_paths = [self.root / 'dataset/raw/item_info.json', 
                      self.root / 'dataset/raw/user_info.json', 
                      self.root / 'dataset/raw/item_share_train_info.json',
                      self.root / 'dataset/raw/item_share_preliminary_test_info.json']
        return _raw_paths
    
    @property
    def user_attrs(self):
        return ["user_gender", "user_age", "user_level"]
    
    @property
    def item_attrs(self):
        return ["cate_id", "cate_level1_id", "brand_id", "shop_id"]



class KGTripletDataset(Dataset):
    def __init__(self,
                 triplets,
                 head_attrs,
                 rel_attrs,
                 tail_attrs=None) -> None:
        super().__init__()
        self.triplets = triplets
        self.head_attrs = head_attrs
        self.rel_attrs = rel_attrs
        self.tail_attrs = tail_attrs
       
    def __len__(self):
        return self.triplets.shape[0]
    
    def __getitem__(self, index):
        if self.tail_attrs is not None:
            return self.triplets[index, :], self.head_attrs[index, :], self.rel_attrs[index, :], self.tail_attrs[index, :]
        else:
            return self.triplets[index, :], self.head_attrs[index, :], self.rel_attrs[index, :]
        
        
class EarlyStoppingMonitor():
    def __init__(self, max_round=10, higher_score=True, tolerance=1e-10) -> None:
        self.max_round = max_round
        self.num_round = 0
        self.best_epoch = 0
        self.best_score = None
        self.higher_score = higher_score
        self.tolerance = tolerance
        
    def check_validation(self, curr_score, curr_epoch):
        if not self.higher_score:
            curr_score *= -1
            
        if self.best_score is None:
            self.best_score = curr_score
        elif (curr_score - self.best_score) / np.abs(self.best_score) > self.tolerance:
            self.best_score = curr_score
            self.num_round = 0
            self.best_epoch = curr_epoch
        else:
            self.num_round += 1
            
        return self.num_round >= self.max_round