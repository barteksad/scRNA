from typing import List
import pandas as pd

class SingleCellMeta():
    def __init__(self, obs_cols: List[str]):
        self.obs_cols = obs_cols

    def get_metadata(self, obs: pd.Series, var: pd.Series) -> str:
        return "\n".join([f"{col}: {obs[col]}" for col in self.obs_cols])