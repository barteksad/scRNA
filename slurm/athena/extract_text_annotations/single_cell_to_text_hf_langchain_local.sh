#!/bin/bash
#SBATCH --account=plgpertext2025-gpu-a100
#SBATCH --job-name=sc_rna
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/sc_rna-%A.log

export HF_HOME="/net/tscratch/people/plgbsadlej/.cache/"

cd /net/tscratch/people/plgbsadlej/scRNA

source ./env/bin/activate

export HYDRA_FULL_ERROR=1

export BATCH_SIZE=64


echo "Running with batch size: $BATCH_SIZE"

srun python src/main.py \
    --config-name single_cell_to_text \
    exp.output_path=data/mouse/generated/single_cell_test.csv \
    exp.model=google/gemma-3-27b-it \
    llm=hf_langchain_local \
    prompt=experimental_1 \
    exp.temperature=1.0 \
    exp.batch_size=$BATCH_SIZE \
    dataset=cell_x_gene \
    dataset.n_files=3 \
    dataset.n_rows_per_file=$(( BATCH_SIZE * 1000 ))