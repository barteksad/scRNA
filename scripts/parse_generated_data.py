import pandas as pd
import json
import glob

files = glob.glob("/net/tscratch/people/plgbsadlej/scRNA/data/mouse/generated/*.csv")
files_ids = [f.split("/")[-1][:-4] for f in files]

for file, file_id in zip(files, files_ids):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    records = []
    current = []

    for line in lines:
        if line.startswith("```json"):
            continue
        elif line.startswith("```"):
            records.append(current)
            current = []
        elif len(current) == 0:
            parts = line.split(",", 2)
            current = parts
        else:
            current[-1] += "\n" + line

    # Step 2: Create DataFrame
    df = pd.DataFrame(records, columns=["source_id", "row_id", "text"])

    def extract_final_response(text):
        try:
            if text and text.strip():
                return json.loads(text[8:])["Final Response"]
        except json.JSONDecodeError as e:
            pass
            # print("JSON error:", e)
        return None  # fallback if empty or invalid

    df["text"] = df["text"].apply(extract_final_response)

    # Optional: clean up columns
    df = df[["source_id", "row_id", "text"]]
    print(file_id, len(df))
    df = df[~df["text"].isnull()]
    print(file_id, len(df))
    print("-" * 20)

    df.to_csv(
        f"/net/tscratch/people/plgbsadlej/scRNA/data/mouse/descriptions/{file_id}.csv",
        index=False,
    )
