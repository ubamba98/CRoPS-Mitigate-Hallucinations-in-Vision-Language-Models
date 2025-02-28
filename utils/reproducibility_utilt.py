import os
import torch
import random
import numpy as np

def set_reproducibility(seed=42, deterministic=True):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    os.environ['PYTHONHASHSEED'] = str(seed)
