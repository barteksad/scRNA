#!/bin/bash
#SBATCH --account=plgpertext2025-gpu-a100
#SBATCH --job-name=sc_rna
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/sc_rna-%A-%a.log
#SBATCH --array=0-5

export HF_HOME="/net/tscratch/people/plgbsadlej/.cache/"

cd /net/tscratch/people/plgbsadlej/scRNA

source ./env/bin/activate

export HYDRA_FULL_ERROR=1

BATCH_SIZES=(16 32 64 128 256 512 1024)

# Get the batch size from the array index
BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Running with batch size: $BATCH_SIZE"
srun python src/main.py \
    --config-name single_cell_to_text \
    exp.model=google/gemma-3-27b-it \
    llm=hf_langchain_local \
    prompt=experimental_1 \
    exp.temperature=1.0 \
    exp.batch_size=$BATCH_SIZE \
    dataset=cell_x_gene \
    dataset.n_rows_per_file=$(( BATCH_SIZE * 3 ))

# exp.model=google/gemma-3-27b-it \