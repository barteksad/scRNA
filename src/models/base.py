from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class BaseSingleCellModel(ABC):
    def __init__():
        pass

    @abstractmethod
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

        pass
