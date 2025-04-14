#!/bin/bash
#SBATCH --account=plgpertext2025-gpu-a100
#SBATCH --job-name=sc_rna
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --output=slurm_logs/sc_rna-%A-%a.log

export HF_HOME="/net/tscratch/people/plgbsadlej/.cache/"

cd /net/tscratch/people/plgbsadlej/scRNA

source ./env/bin/activate

export HYDRA_FULL_ERROR=1

export BATCH_SIZE=256


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

IDS=(
060e9a8d-e5c4-4f29-a60d-febf5c545704
2d50c165-da20-47c4-9cbe-914f8a8b3211
69a7c0d4-c8a5-4ad7-963c-ba49ed703913
891afa6c-0b3d-44a1-9448-31cdea6bddca
d3874464-500b-4815-a100-f2cc9000e64d
96e83818-c3aa-422f-93eb-a4e08366e192
0f691454-f7f1-41f4-ba3d-544905a74f57
29e35613-d182-4016-bb5d-4df14bcee128
3ac354e6-1b32-402f-a0de-d790d37f962d
8efaf5ba-9121-451a-a4d4-d63add2404d8
9869c8ab-662a-4c10-96c2-a4dd96334e51
aafe93a0-85d1-45f3-b081-0aa36f1738e4
b6fff8a8-b297-4715-b041-43d23a4e0474
b752cc64-2ea2-4dc1-ade2-2c8e8fea949d
f6f5b47a-f18b-4756-9eac-d247f741ea3c
c64ea37e-91a7-4670-870e-eae3a709cb12
)

sleep $(( SLURM_ARRAY_TASK_ID * 3 ))

FILE_ID=${IDS[$SLURM_ARRAY_TASK_ID]}

srun python src/main.py \
    --config-name single_cell_to_text \
    exp.output_path=data/mouse/generated/ \
    exp.model=google/gemma-3-27b-it \
    llm=hf_langchain_local \
    prompt=experimental_1 \
    exp.temperature=1.0 \
    exp.batch_size=$BATCH_SIZE \
    dataset=cell_x_gene \
    exp.file_id=$FILE_ID

# python src/main.py \
#     --config-name single_cell_to_text \
#     exp.output_path=data/mouse/generated/ \
#     exp.model=gemma-3-27b-it \
#     llm=google \
#     prompt=experimental_1 \
#     exp.temperature=1.0 \
#     exp.batch_size=$BATCH_SIZE \
#     dataset=cell_x_gene \
#     exp.file_id=$FILE_ID