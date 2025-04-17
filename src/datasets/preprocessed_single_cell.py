import glob
import os
import pandas as pd

import torch
from torch.utils.data import Dataset


class PreprocessedCellDataset(Dataset):
    def __init__(
        self,
        data_dir,
        obs_cols,
        cell_tokenizer,
        text_tokenizer,
        max_cell_tokens,
        max_text_tokens,
    ):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.cell_tokenizer = cell_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_cell_tokens = max_cell_tokens
        self.max_text_tokens = max_text_tokens

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)

        _, tokenized_cells = self.cell_tokenizer.tokenize_single_cell(
            data["x"],
            pd.DataFrame([data["obs"]]),
            pd.DataFrame({"gene_name": data["var"]}),
        )
        positional_encoding = torch.tensor(
            tokenized_cells[0].fillna(0).astype(int).values,
            dtype=torch.int32,
        )

        # truncate
        cell_tokens = positional_encoding[
            : min(len(positional_encoding), self.max_cell_tokens)
        ]

        encoding = self.text_tokenizer(
            data["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens,
        )

        return {
            "cell_tokens": cell_tokens,
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
        }
