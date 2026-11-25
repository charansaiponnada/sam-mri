import random

import numpy as np

SEQS = ["t1", "t1ce", "t2", "flair"]
SEQ_IDX = {s: i for i, s in enumerate(SEQS)}
REGIONS = ["WT", "TC", "ET"]


def region_mask(seg, region):
    """seg uses canonical labels {0, 1=NCR, 2=ED, 4=ET}. Works on arrays of any dim."""
    if region == "WT":
        return seg > 0
    if region == "TC":
        return (seg == 1) | (seg == 4)
    if region == "ET":
        return seg == 4
    raise ValueError(f"unknown region {region}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
