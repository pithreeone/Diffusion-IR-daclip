import random
import numpy as np
import torch

def seed_everything(SEED=42):
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False