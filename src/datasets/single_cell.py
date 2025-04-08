import glob
import os
import pandas as pd
import numpy as np

from typing import List
from scanpy import read_h5ad
from torch.utils.data import Dataset


class SingleCellDataset(Dataset):
    def __init__(
        self,
        h5ad_dir: str,
        obs_cols: List[str],
        description_dir: str,
    ):
        self.source_id2h5ad_files = dict([
            (file.split("/")[-1][:-4], read_h5ad(file))
            for file in glob.glob(os.path.join(os.getcwd()[:-3] + h5ad_dir, "*"))
        ])
        self.source_id2description_file = dict(
            [
                (file.split("/")[-1][:-4], pd.read_csv(file))
                for file in glob.glob(
                    os.path.join(os.getcwd()[:-3] + description_dir, "*")
                )
            ]
        )

        lenghts_array = [len(df) for df in self.source_id2description_file.values()]
        self.length = sum(lenghts_array)
        self.row_idx2_source_id = [
            (s, file_id)
            for s, file_id in zip(
                np.cumsum(lenghts_array), self.source_id2description_file.keys()
            )
        ]

        self.obs_cols = obs_cols

    def __len__(self):
        return self.length
    
    def __getitem__(self, row_idx):
        source_id = self.row_idx2_source_id[0][1]
        
        i = 1
        to_subtract = 0
        while row_idx > self.row_idx2_source_id[i][0]:
            to_subtract += self.row_idx2_source_id[i - 1][0]
            source_id = self.row_idx2_source_id[i][1]
            i += 1
        
        row_idx -= to_subtract
        matching_descriptions_file = self.source_id2description_file[source_id]
        matching_descriptions = matching_descriptions_file[
            (matching_descriptions_file["source_id"] == source_id)
        ].iloc[row_idx]

        if len(matching_descriptions) == 0:
            return None
        
        data = self.source_id2h5ad_files[source_id][matching_descriptions["row_id"]]
        obs = data.obs.iloc[row_idx]
        var = data.var
        x = data.X[row_idx]
        
        return {
            "cell_data": (x, obs, var),
            "text": matching_descriptions.iloc[0]["text"],
        }