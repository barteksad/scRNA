#!/bin/bash
#SBATCH --account=plgpertext2025-gpu-a100
#SBATCH --job-name=sc_rna_multigpu
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/sc_rna-%A.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

cd /net/tscratch/people/plgbsadlej/scRNA

export HYDRA_FULL_ERROR=1

export HF_HOME="/net/tscratch/people/plgbsadlej/.cache/"
export TOKENIZERS_PARALLELISM=false

source ./env/bin/activate

export WANDB_CACHE_DIR="/net/tscratch/people/plgbsadlej/.cache/wandb"
export WANDB_ARTIFACT_DIR="/net/tscratch/people/plgbsadlej/.cache/wandb/artifacts"

export PYTHONPATH="/net/tscratch/people/plgbsadlej/scRNA"
export OPENAI_API_KEY="dummy key"

# Set up distributed training environment
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500

# Set environment variables for better distributed training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Run distributed training using torchrun
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT src/clip.py