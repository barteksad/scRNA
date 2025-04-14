srun --account=plgpertext2025-gpu-a100 --job-name=test --partition=plgrid-gpu-a100 --mem=20G --gres=gpu:1 --time=1:00:00 --pty bash
srun --account=plgpertext2025-gpu-a100 --job-name=test --partition=plgrid --mem=10G --time=1:00:00 --pty bash

# srun --pty --overlap --jobid  1343903 bash