from scanpy import read_h5ad

data = read_h5ad("data/mouse/datasets/scCompass/h5ad?name=219dc3bf62144d901d48341d5002df99")

data.obs_keys()

gene_expression_matrix = data.X[:1]
obs = data.obs.iloc[:1]
var = data.var

data.obs # -> [11123 rows x 11 columns]

data.var # -> [14549 rows x 10 columns]
# gene_name -> GENE_MAPPING_FILE -> ensemle_id
data.X # -> (11123, 14549)



GENE_MAPPING_FILE = "data/mouse/weights/mouse-Geneformer/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"
TOKEN_DICTIONARY_FILE = "data/mouse/weights/mouse-Geneformer/MLM-re_token_dictionary_v1.pkl"
GENE_MEDIAN_FILE = "data/mouse/weights/mouse-Geneformer/mouse_gene_median_dictionary.pkl"

with open(GENE_MAPPING_FILE, "rb") as f:
    gene_mapping_dict = pickle.load(f)
with open(GENE_MEDIAN_FILE, "rb") as f:
    gene_median_dict = pickle.load(f)
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    token_dict = pickle.load(f)

genelist_dict = dict(zip(gene_keys, [True] * len(gene_keys)))