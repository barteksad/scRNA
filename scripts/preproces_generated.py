import os
import glob
import torch
import pandas as pd
from scanpy import read_h5ad
from omegaconf import OmegaConf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def process_description_file(src_id, desc_path, h5ad_dir, output_dir, obs_cols):
    try:
        h5ad_path = os.path.join(h5ad_dir, f"{src_id}.h5ad")
        if not os.path.exists(h5ad_path):
            print(f"Missing h5ad file for {src_id}")
            return 0

        h5ad = read_h5ad(h5ad_path)
        desc_df = pd.read_csv(desc_path)

        count = 0
        for i, row in desc_df.iterrows():
            row_id = row.get("row_id")

            try:
                adata = h5ad[row_id]
                x_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

                # Extract only the relevant observation data for this cell
                obs_data = {}
                for col in adata.obs.columns:
                    obs_data[col] = adata.obs[col]

                record = {
                    "x": x_data,
                    "obs": obs_data,  # Only save the relevant observation data
                    "var": adata.var[["gene_name"]].values.flatten().tolist()
                    if "gene_name" in adata.var.columns
                    else adata.var[["feature_name"]].values.flatten().tolist(),
                    "text": row.get("text", ""),
                }

                torch.save(record, os.path.join(output_dir, f"{src_id}_{i}.pt"))
                count += 1

            except Exception as e:
                print(f"Failed to process {src_id}:{row_id} — {e}")
                continue

        return count
    except Exception as e:
        print(f"Error in {src_id}: {e}")
        return 0


def unpack_and_process(task, fn):
    return fn(*task)


def parallel_preprocess(
    h5ad_dir, description_dir, output_dir, obs_cols, num_workers=None
):
    os.makedirs(output_dir, exist_ok=True)
    desc_files = glob.glob(os.path.join(description_dir, "*.csv"))

    # (src_id, path) pairs
    tasks = [(os.path.basename(f)[:-4], f) for f in desc_files]

    # process_description_file(
    #     tasks[0][0],
    #     tasks[0][1],
    #     h5ad_dir=h5ad_dir,
    #     output_dir=output_dir,
    #     obs_cols=obs_cols,
    # )

    fn = partial(
        process_description_file,
        h5ad_dir=h5ad_dir,
        output_dir=output_dir,
        obs_cols=obs_cols,
    )

    with Pool(processes=num_workers or cpu_count()) as pool:
        # inside parallel_preprocess()
        results = list(
            tqdm(
                pool.imap_unordered(partial(unpack_and_process, fn=fn), tasks),
                total=len(tasks),
            )
        )

    print(f"Total processed samples: {sum(results)}")


if __name__ == "__main__":
    config = OmegaConf.load("configs/dataset/cell_x_gene.yaml")

    h5ad_dir = config.h5ad_dir
    description_dir = config.description_dir
    obs_cols = config.obs_cols
    output_dir = "data/mouse/preprocessed"

    parallel_preprocess(
        h5ad_dir=h5ad_dir,
        description_dir=description_dir,
        output_dir=output_dir,
        obs_cols=obs_cols,
        num_workers=None,  # Defaults to CPU count
    )
