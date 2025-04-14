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

export BATCH_SIZE=512


echo "Running with batch size: $BATCH_SIZE"

IDS=(
fd0e720f-eb2d-49d9-8a3d-0c6001789ed9
cb26dbc6-9eb6-4d26-9561-3f1f851b89b9
242b59a3-7398-4f0d-89fc-73cbc886d1b9
8c64b76f-6798-43b4-9e22-a4c69be77325
fd955e8d-e90a-42b2-828e-d5a1a4ff4100
cd6c6c36-d483-4025-8840-9d0b766c40ee
77d6a75a-1f9e-4a61-9804-c88cedd11df0
53ccd1cf-3c64-4a60-b790-44d2c82bb0e7
94a836ba-0a79-4f4a-9931-aa2a2cb7a22e
fc0fcab1-bd87-4a5f-a895-ad95fe9aa7a9
ec129423-41cd-4ecf-997f-8a68bb479d9f
d74d2280-1815-4fb0-8460-f9be2d7100fa
18fb432a-be41-4677-9396-e1680a0852bf
)

srun python src/main.py \
    --config-name single_cell_to_text \
    exp.output_path=data/mouse/generated/single_cell_test.csv \
    exp.model=google/gemma-3-27b-it \
    llm=hf_langchain_local \
    prompt=experimental_1 \
    exp.temperature=1.0 \
    exp.batch_size=$BATCH_SIZE \
    dataset=cell_x_gene \
    exp.file_id=