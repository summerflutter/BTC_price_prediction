import os, random, numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def pick_device(pref: str = "auto"):
    try:
        if pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return pref
    except Exception:
        return "cpu"
