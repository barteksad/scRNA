SSH to Athena
```
login@athena.cyfronet.pl
```
Athena interactive session
```
srun --account=plgpertext2025-gpu-a100 --job-name=test --partition=plgrid-gpu-a100 --gres=gpu:1 --time=1:00:00 --pty bash

```

Store all files under
```
/net/tscratch/people/USER
```
because $HOME is small


Create env and install dependencies
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Copy `.env.template` as `.env` and fill in the values

# NEW Data
you can download the data from google drive
```
https://drive.google.com/file/d/1-2xpFnik-0pPGFkpxw9hym05zLtwYlB4/view?usp=sharing
```

```
Sample text data csv file
https://docs.google.com/spreadsheets/d/1jr-df0TZZsK165rtTvKoUmQSGxUf6WEq_Ccqw0mevwA/edit?usp=sharing
```