#!/bin/bash
#SBATCH --account=plgpertext2025-gpu-a100
#SBATCH --job-name=sc_rna
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/sc_rna-%A.log

cd /net/tscratch/people/plgbsadlej/scRNA

export HYDRA_FULL_ERROR=1

export HF_HOME="/net/tscratch/people/plgbsadlej/.cache/"
export TOKENIZERS_PARALLELISM=false

source ./env/bin/activate

export WANDB_CACHE_DIR="/net/tscratch/people/plgbsadlej/.cache/wandb"
export WANDB_ARTIFACT_DIR="/net/tscratch/people/plgbsadlej/.cache/wandb/artifacts"

export PYTHONPATH="/net/tscratch/people/plgbsadlej/scRNA"
export OPENAI_API_KEY="dummy key"
srun python src/clip.py