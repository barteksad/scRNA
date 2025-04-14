import glob
import os
from typing import List

from scanpy import read_h5ad
from torch.utils.data import Dataset


class SingleCellDataset(Dataset):
    def __init__(self, h5ad_dir: str, file_id: str, obs_cols: List[str]):
        self.h5ad_dir = h5ad_dir
        self.file_id = file_id
        self.obs_cols = obs_cols
        self.file = os.path.join(h5ad_dir, f"{file_id}.h5ad")
        self.data = read_h5ad(self.file)

    def __len__(self):
        return self.data.X.shape[0]

    def __getitem__(self, row_idx: int):
        obs = self.data.obs.iloc[row_idx]  # [self.obs_cols]
        var = self.data.var
        x = self.data.X[row_idx]

        return x, obs, var, self.file_id, row_idx
