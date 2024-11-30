# ruff: noqa: F401
import warnings
from pathlib import Path
import sys

warnings.filterwarnings(
    "ignore", message=".*The 'nopython' keyword.*"
)  # noqa # isort:skip

# GENE_MEDIAN_FILE = Path(__file__).parent / "gene_median_dictionary_gc95M.pkl"
# TOKEN_DICTIONARY_FILE = Path(__file__).parent / "token_dictionary_gc95M.pkl"
# ENSEMBL_DICTIONARY_FILE = Path(__file__).parent / "gene_name_id_dict_gc95M.pkl"
# ENSEMBL_MAPPING_FILE = Path(__file__).parent / "ensembl_mapping_dict_gc95M.pkl"

# BASE_PATH = "/Users/barteksadlej/others/UW/TML/data/mouse/weights/mouse-Geneformer"

# GENE_MEDIAN_FILE = Path(BASE_PATH) / "mouse_gene_median_dictionary.pkl"
# TOKEN_DICTIONARY_FILE = Path(BASE_PATH) / "MLM-re_token_dictionary_v1.pkl"
# ENSEMBL_DICTIONARY_FILE = (
#     Path(BASE_PATH) / "MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"
# )
# ENSEMBL_MAPPING_FILE = Path(BASE_PATH) / "MLM-re_All_mouse_tokenize_dataset_length.pkl"

GENE_MEDIAN_FILE = None
TOKEN_DICTIONARY_FILE = None
ENSEMBL_DICTIONARY_FILE = None
ENSEMBL_MAPPING_FILE = None

from . import (
    collator_for_classification,
    emb_extractor,
    in_silico_perturber,
    in_silico_perturber_stats,
    pretrainer,
    tokenizer,
)
from .collator_for_classification import (
    DataCollatorForCellClassification,
    DataCollatorForGeneClassification,
)
from .emb_extractor import EmbExtractor, get_embs
from .in_silico_perturber import InSilicoPerturber
from .in_silico_perturber_stats import InSilicoPerturberStats
from .pretrainer import GeneformerPretrainer
from .tokenizer import TranscriptomeTokenizer

from . import classifier  # noqa # isort:skip
from .classifier import Classifier  # noqa # isort:skip

from . import mtl_classifier  # noqa # isort:skip
from .mtl_classifier import MTLClassifier  # noqa # isort:skip
