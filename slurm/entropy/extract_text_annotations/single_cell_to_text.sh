#!/bin/bash
#SBATCH --job-name=sc_rna
#SBATCH --partition=common
#SBATCH --qos=4gpu1h
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/sc_rna-%A.log

source ./env/bin/activate

export HYDRA_FULL_ERROR=1

srun python src/main.py \
    --config-name single_cell_to_text \
    exp.temperature=1.0 \
    dataset.n_rows_per_file=1 \
    dataset.h5ad_dir=/Users/barteksadlej/others/UW/ZPML/scRNA/data/mouse/datasets/scCompass \
    model.gene_mapping_file=/Users/barteksadlej/others/UW/ZPML/scRNA/data/mouse/weights/mouse-Geneformer/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl \
    model.gene_median_file=/Users/barteksadlej/others/UW/ZPML/scRNA/data/mouse/weights/mouse-Geneformer/MLM-re_token_dictionary_v1.pkl \
    model.token_dictionary_file=/Users/barteksadlej/others/UW/ZPML/scRNA/data/mouse/weights/mouse-Geneformer/mouse_gene_median_dictionary.pkl
