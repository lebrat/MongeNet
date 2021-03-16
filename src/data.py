import torch
from torch.utils.data import Dataset
import numpy as np

class TriangleSampling(Dataset):
    # dataset reader for triangles and optimal point clouds approximations
    def __init__(self, xpath, ypath):
        self.xlist = xpath
        self.ylist = ypath

    def __len__(self):
        return len(self.xlist)

    def __getitem__(self, index):
        X, Y = np.load(self.xlist[index]).astype(np.float), np.load(self.ylist[index]).astype(np.float)
        # reshape to better readability and convert to Cartesian coordinates
        Y = Y.transpose().dot(X)
        return X, Y


class CombinedIterator:
    # combine multiples triangle sampling data-loaders
    def __init__(self, dataloaders, probs=None):
        assert probs is None or len(dataloaders) == probs.numel()
        self.dataloaders = [iter(dl) for dl in dataloaders]
        self.probs = np.ones(len(dataloaders)) if probs is None else probs
        self.probs = self.probs / (self.probs.sum() + 1e-12)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                ds_idx = int(np.random.choice(len(self.probs), 1, p=self.probs))        
                next_batch = next(self.dataloaders[ds_idx])             
                return next_batch

            except StopIteration:               
                self.probs[ds_idx] = 0.0
                self.probs = self.probs / (self.probs.sum() + 1e-12)
            
            if self.probs.sum() == 0:
                raise StopIteration