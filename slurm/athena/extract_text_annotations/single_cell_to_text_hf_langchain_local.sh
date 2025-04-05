#!/bin/bash
#SBATCH --account=plgpertext2025-gpu-a100
#SBATCH --job-name=sc_rna
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/sc_rna-%A.log

cd /net/tscratch/people/plgbsadlej/scRNA

source ./env/bin/activate

export HYDRA_FULL_ERROR=1

srun python src/main.py \
    --config-name single_cell_to_text \
    llm=hf_langchain_local \
    exp.model=google/gemma-3-27b-it \
    prompt=experimental_1 \
    exp.temperature=1.0 \
    dataset=cell_x_gene \
    dataset.n_rows_per_file=1 \
    dataset.n_files=1
