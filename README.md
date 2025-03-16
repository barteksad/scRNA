To run interactive session on entropy. You may need to change qos to XgpuYh depending on the account you have.
```
srun --job-name=test --partition=common --qos=4gpu1h --time=1:00:00 --pty bash
```

Create env and install dependencies
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Copy `.env.template` as `.env` and fill in the values

# NEW Data
you can download the data from google drive
```
https://drive.google.com/file/d/1cGZDNnsqCe0GGK5Qs80fV4kccc9k2-uW/view?usp=sharing
```

# OLD
The `data` folder is linked to shared folder `local_storage_1` which is available in the file system on entrophy for slurm sessions (so not on access node) under `/local_storage_1/bs429589/sc_rna/data`

Run single cell text annotations with llm
```
sbatch slurm/entropy/extract_text_annotations/single_cell_to_text.sh
```
with openai o1 model
```
sbatch slurm/entropy/extract_text_annotations/single_cell_to_text_openai_o1.sh
```

