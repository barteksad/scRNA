import glob
import os
import pandas as pd
import numpy as np

from typing import List, Dict, Optional
from scanpy import read_h5ad
from torch.utils.data import Dataset


class SingleCellDataset(Dataset):
    def __init__(
        self,
        h5ad_dir: str,
        obs_cols: List[str],
        description_dir: str,
    ):
        # Load only the first 2 h5ad files (might be for testing purposes)
        self.source_id2h5ad_files = {
            os.path.basename(file)[:-5]: read_h5ad(file)
            for file in glob.glob(os.path.join(h5ad_dir, "*"))
        }

        # Fix the nested glob.glob issue and properly load description files
        self.source_id2description_file = {
            os.path.basename(file)[:-4]: pd.read_csv(file)
            for file in glob.glob(os.path.join(description_dir, "*"))
        }

        # Calculate dataset length and index mapping
        self.lengths_array = [
            len(df) for df in self.source_id2description_file.values()
        ]
        self.length = sum(self.lengths_array)

        # Build a mapping from row indices to source IDs and cumulative indices
        cumulative_sums = np.cumsum([0] + self.lengths_array[:-1])
        self.row_idx2source_id = list(
            zip(cumulative_sums, self.source_id2description_file.keys())
        )

        self.obs_cols = obs_cols

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, row_idx: int) -> Optional[Dict]:
        if row_idx < 0 or row_idx >= self.length:
            raise IndexError(
                f"Index {row_idx} out of bounds for dataset of length {self.length}"
            )

        # Find the correct source_id for this row_idx
        source_id = None
        idx_in_source = row_idx

        for i, (start_idx, src_id) in enumerate(self.row_idx2source_id):
            if i < len(self.row_idx2source_id) - 1:
                next_start_idx = self.row_idx2source_id[i + 1][0]
                if start_idx <= row_idx < next_start_idx:
                    source_id = src_id
                    idx_in_source = row_idx - start_idx
                    break
            else:
                # Last segment
                source_id = src_id
                idx_in_source = row_idx - start_idx

        # Get the description file for this source_id
        description_file = self.source_id2description_file.get(source_id)
        if description_file is None:
            return None

        # Get the matching description
        if idx_in_source >= len(description_file):
            return None

        matching_description = description_file.iloc[idx_in_source]

        # Get the corresponding data
        h5ad_file = self.source_id2h5ad_files.get(source_id)
        if h5ad_file is None:
            return None

        # Extract the cell data using the row_id from the description
        row_id = matching_description.get("row_id")
        if row_id is None:
            return None

        try:
            data = h5ad_file[row_id]
            obs = data.obs
            var = data.var
            x = data.X

            return {
                "cell_data": (x, obs, var),
                "text": matching_description.get("text", ""),
            }
        except (KeyError, IndexError):
            return None
