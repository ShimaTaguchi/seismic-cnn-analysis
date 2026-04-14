import torch
import numpy as np
import random
import logging

# ImageNet 正規化
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_device() -> torch.device:
    """デバイス（GPU/CPU）"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def set_seed(seed: int) -> None:
    """乱数シードを固定する"""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False