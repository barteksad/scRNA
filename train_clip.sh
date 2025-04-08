#!/bin/bash
#SBATCH --job-name=clip_genomics
#SBATCH --partition=common
#SBATCH --gres=gpu:titanx:1
#SBATCH --time=00:30:00
#SBATCH --output=clip_genomics_log.txt


export HYDRA_FULL_ERROR=1
source .venv/bin/activate
PYTHONPATH=/home/pablo2811/scRNA /home/pablo2811/scRNA/.venv/bin/python /home/pablo2811/scRNA/src/clip.py
