import pickle
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import BaseSingleCellModel


def rank_genes(gene_vector, gene_tokens, gene_names):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector)[:2048]

    tokens = gene_tokens[sorted_indices]
    gene_names = gene_names.values[sorted_indices]

    df = pd.Series(gene_names, index=tokens)

    return df


class MouseFormer(BaseSingleCellModel):
    def __init__(
        self,
        gene_mapping_file: str,
        gene_median_file: str,
        token_dictionary_file: str,
        target_sum=10_000,
        chunk_size=512,
    ):
        self.target_sum = target_sum
        self.chunk_size = chunk_size

        with open(gene_mapping_file, "rb") as f:
            self.gene_mapping_dict = pickle.load(f)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)
        with open(token_dictionary_file, "rb") as f:
            self.token_dict = pickle.load(f)

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def tokenize_single_cell(
        self, gene_expression_matrix, obs: pd.DataFrame, var: pd.DataFrame
    ) -> List[pd.Series]:
        """
        Tokenize a single cell gene expression matrix

        Args:
        gene_expression_matrix: N cells x n genes of expression values
        var: DataFrame containing gene metadata

        Returns:
        List of tokenized cells where each element is pandas series with index as token integer and value being gene name
        """
        key_to_use = "gene_name" if "gene_name" in var.columns else "feature_name"
        ensemble_ids = var[key_to_use].map(self.gene_mapping_dict)

        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in ensemble_ids]
        )[0]
        norm_factor_vector = np.array(
            [self.gene_median_dict[i] for i in ensemble_ids[coding_miRNA_loc]]
        )
        coding_miRNA_ids = ensemble_ids.iloc[coding_miRNA_loc]
        coding_miRNA_tokens = np.array([self.token_dict[i] for i in coding_miRNA_ids])

        tokenized_cells = []

        for i in range(0, gene_expression_matrix.shape[0], self.chunk_size):
            idx = slice(i, i + self.chunk_size)

            col_to_use = "n_genes" if "n_genes" in obs[idx] else "nFeature_RNA"
            if col_to_use not in obs[idx]:
                n_counts = 2500
            else:
                n_counts = obs[idx][col_to_use]
            X_view = gene_expression_matrix[idx, coding_miRNA_loc]
            X_norm = X_view / n_counts * self.target_sum / norm_factor_vector
            X_norm = sp.csr_matrix(X_norm)

            tokenized_cells += [
                rank_genes(
                    X_norm[i].data,
                    coding_miRNA_tokens[X_norm[i].indices],
                    var[key_to_use],
                )
                for i in range(X_norm.shape[0])
            ]

        return tokenized_cells
