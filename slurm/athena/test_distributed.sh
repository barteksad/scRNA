#!/bin/bash
#SBATCH --account=plgpertext2025-gpu-a100
#SBATCH --job-name=test_distributed
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/test_distributed-%A.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

cd /net/tscratch/people/plgbsadlej/scRNA

source ./env/bin/activate

export PYTHONPATH="/net/tscratch/people/plgbsadlej/scRNA"

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501

echo "Starting distributed training test..."
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Run distributed training test
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT test_distributed.py

echo "Test completed!" 