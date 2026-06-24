# utils/set_seed.py
import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    设置随机种子以确保实验的可重复性

    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用CUDA（GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

        # 添加以下设置以获得更好的可重复性（可能会降低性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 设置Python哈希种子（用于dict等）
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"随机种子已设置为: {seed}")