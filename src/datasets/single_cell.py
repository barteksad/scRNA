from typing import List
from scanpy import read_h5ad
import glob
import os
from torch.utils.data import Dataset

class SingleCellDataset(Dataset):

    def __init__(self, h5ad_dir: str, n_files: int, n_rows_per_file: int, obs_cols: List[str]):
        self.h5ad_dir = h5ad_dir
        self.n_files = n_files
        self.n_rows_per_file = n_rows_per_file
        self.obs_cols = obs_cols
        self.files = glob.glob(os.path.join(h5ad_dir, "*"))

    def __len__(self):
        return self.n_files * self.n_rows_per_file
    
    def __getitem__(self, idx: int):
        file_idx = idx // self.n_rows_per_file
        row_idx = idx % self.n_rows_per_file
        data = read_h5ad(self.files[file_idx])
        obs = data.obs.iloc[row_idx][self.obs_cols]
        var = data.var
        x = data.X[row_idx]
        return x, obs, var