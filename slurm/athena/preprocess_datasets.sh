#!/bin/bash
#SBATCH --account=plgpertext2025-gpu-a100
#SBATCH --job-name=sc_rna
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/sc_rna-%A.log


cd /net/tscratch/people/plgbsadlej/scRNA

source ./env/bin/activate

srun python scripts/preproces_generated.py