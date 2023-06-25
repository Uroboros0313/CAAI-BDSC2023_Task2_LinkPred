import os
from pathlib import Path


ROOT_DIR = os.path.dirname(__file__)
DATACENTER_SAVE_PATH = Path(ROOT_DIR) / 'dataset/processed/kg_datacenter.pkl'
MODEL_SAVE_DIR = Path(ROOT_DIR) / 'saved_models'

BATCH_SIZE = 4096 * 12
MAX_EPOCHS = 100
EMBEDDING_DIM = 256
DEVICE = 'cuda'
SEED = 1008
MARGIN = 10
TOPK = 5

DISMULT_EPOCHS = 23
DISMULT_LR = 0.005
