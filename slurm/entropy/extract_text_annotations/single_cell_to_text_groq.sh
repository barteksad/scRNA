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
    llm=groq \
    exp.model=qwen-qwq-32b \
    prompt=experimental_1 \
    exp.temperature=1.0 \
    dataset=cell_x_gene \
    dataset.n_rows_per_file=10 \
    dataset.n_files=1
