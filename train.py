import pickle
import time
import logging
import random
import json

import faiss
import numpy as np
import torch as th
import torch.optim as optim
from tqdm import tqdm

from constant import *
from utils import KGDataCenter, EarlyStoppingMonitor
from models.dismult import DisMultGES


IS_DEV = False

if not os.path.exists(Path(ROOT_DIR) / 'log'):
    os.makedirs(Path(ROOT_DIR) / 'log')
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DisMultGES')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(Path(ROOT_DIR) / 'log/{}.log'.format(str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(int(time.time()))))))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def set_rand_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    

def generate_recall_result(model, test_dataset, topK=5, model_id=0):
    tst_tuplets, tst_ha, tst_ra = test_dataset
    with th.no_grad():
        model.eval()
        embeddings = model.ent_table.weight.data.cpu().numpy()
        indexerIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        indexerIP.add(embeddings)
        
        query = model.infer(tst_tuplets.to(DEVICE), 
                            tst_ra[:, :].to(DEVICE)).cpu().numpy()
    
    distance, topKindex = indexerIP.search(query, topK)
    submit = []
    for i in tqdm(range(tst_tuplets.shape[0])):
        topK_list = list(topKindex[i, :])
        candidate_voter_list = [kg_center.usernum2id[top_voter_index] for top_voter_index in topK_list]
        submit.append({'triple_id': str('%06d' % i), 'candidate_voter_list': candidate_voter_list})

    submission_save_dir = Path('./dataset/submit')
    if not os.path.exists(submission_save_dir):
        os.mkdir(submission_save_dir)
    
    with open(submission_save_dir / f'submit_DisMult_GES_{model_id}.json', 'w') as f:
        json.dump(submit, f)


def validation_mrr(model, valid_dataset):
    (val_triplets, val_ha, val_ra, val_ta) = valid_dataset
    with th.no_grad():
        model.eval()
        embeddings = model.ent_table.weight.data.cpu().numpy()
        indexerIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        indexerIP.add(embeddings)
        
        query = model.infer(val_triplets[:, :2].to(DEVICE), 
                            val_ra[:, :].to(DEVICE)).cpu().numpy()
        val_tails = val_triplets[:, 2].cpu().numpy().tolist()
    
    distance, topKindex = indexerIP.search(query, TOPK)

    mrr_score = 0.0
    hit_5user = 0.0
    for i in range(len(val_tails)):
        voter = val_tails[i]
        topK_list = list(topKindex[i, :])
        if voter in topK_list:
            rank = topK_list.index(voter)
            mrr_score += 1.0 / (rank + 1)
            hit_5user += 1.0
    
    mrr_score /= len(val_tails)
    hit_5user /= len(val_tails)
    
    return mrr_score, hit_5user 


def train(model_id=0, lr_noise=0.0, epoch_noise=0):
    model = DisMultGES(ent_vocab_size=kg_center.num_user,
                            rel_vocab_size=kg_center.num_item,
                            embedding_dim=EMBEDDING_DIM,
                            rel_attrs_enc_map=kg_center.item_attrs_enc_map,
                            margin=MARGIN,
                            device=DEVICE).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=DISMULT_LR + lr_noise)
    earlystopper = EarlyStoppingMonitor()
    
    for epoch in range(DISMULT_EPOCHS + epoch_noise):
        l_sum = 0.0
        for trn_batch, neg_trn_batch in tqdm(zip(train_loader, neg_train_loader)):
            trn_batch_triplets, trn_batch_ha, trn_batch_ra, trn_batch_ta = trn_batch
            ntrn_batch_triplets, ntrn_batch_ha, ntrn_batch_ra, ntrn_batch_ta = neg_trn_batch
            
            pos_dis = model(trn_batch_triplets.to(DEVICE), trn_batch_ra.to(DEVICE))
            neg_dis = model(ntrn_batch_triplets.to(DEVICE), ntrn_batch_ra.to(DEVICE))
            optimizer.zero_grad()
            l = model.criterion(pos_dis, neg_dis)
            l.backward()
            optimizer.step()
            
            l_sum += l.item()
        
        gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
        logger.info(' Epoch: {} || GPU Occupy: {:.4f}.MiB || Train Loss: {:.4f} ||'.\
            format(epoch, gpu_mem_alloc, l_sum))
        
        if IS_DEV:
            mrr_score, hit_5user = validation_mrr(model, valid_dataset)
            logger.info(' Val_MRR: {:.6f} || HIT@5: {:.6f} ||'.format(mrr_score, hit_5user))

            if not earlystopper.check_validation(mrr_score, epoch):
                if epoch == earlystopper.best_epoch:
                    with open(MODEL_SAVE_DIR / f'dismultGES_{model_id}.pkl', 'wb') as f:
                        pickle.dump(model, f)
            else:
                logger.info("Model Not Improved For {} epochs, Best Epoch is {}".\
                    format(earlystopper.max_round, earlystopper.best_epoch))
                break
        else:
            with open(MODEL_SAVE_DIR / f'dismultGES_{model_id}.pkl', 'wb') as f:
                pickle.dump(model, f)
                
    return model

            
        

if __name__ == '__main__':
    if os.path.exists(DATACENTER_SAVE_PATH):
        with open(DATACENTER_SAVE_PATH, 'rb') as f:
            kg_center = pickle.load(f)
    else:
        kg_center = KGDataCenter(ROOT_DIR)
        with open(DATACENTER_SAVE_PATH, 'wb') as f:
            pickle.dump(kg_center, f)
    
            
    lr_noise_list = np.arange(-0.001, 0.0015, 0.00025)
    epoch_noise_list = list(range(5, 0, -1))
    for model_id in range(20):
        set_rand_seed(SEED + model_id - 2)
        
        train_loader, neg_train_loader = kg_center.generate_train_dataloader(is_dev=IS_DEV)
        valid_dataset = kg_center.generate_valid_dataset()
        test_dataset = kg_center.generate_test_dataset()
        
        if not os.path.exists(MODEL_SAVE_DIR / f'dismultGES_{model_id}.pkl'):
            model=train(model_id, lr_noise=lr_noise_list[model_id%10], epoch_noise=epoch_noise_list[model_id%5])
            generate_recall_result(model, test_dataset, topK=10)
        else:
            with open(MODEL_SAVE_DIR / f'dismultGES_{model_id}.pkl', 'rb') as f:
                model = pickle.load(f)
            logger.info(f'Current model id: {model_id}')
            generate_recall_result(model, test_dataset, topK=10, model_id=model_id)