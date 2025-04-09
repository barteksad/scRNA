import logging
import torch
from torch.utils.data import Dataset, DataLoader
import time

import weave
from hydra.utils import instantiate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BaseChatPromptTemplate
from omegaconf import DictConfig
from tqdm import tqdm
import utils
from models import BaseSingleCellModel

log = logging.getLogger(__name__)


def get_components(config):
    dataset = instantiate(config.dataset)
    metadata = instantiate(config.metadata)
    prompt: BaseChatPromptTemplate = instantiate(config.prompt)
    llm = instantiate(config.llm) | StrOutputParser()
    model: BaseSingleCellModel = instantiate(config.model)

    return dataset, metadata, prompt, llm, model


class AnnotationDataset(Dataset):
    def __init__(self, dataset, metadata, model, prompt, config):
        self.dataset = dataset
        self.metadata = metadata
        self.model = model
        self.prompt = prompt
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, obs, var, source_id, row_idx = self.dataset[idx]
        
        if len(x.shape) < 2:
            x = x[None, ...]
            
        tokenized_cell = self.model.tokenize_single_cell(x, obs, var)
        meta_text = self.metadata.get_metadata(obs, var, source_id)
        
        if "top_k_genes" in self.prompt.input_variables:
            messages_list = self.prompt.format(
                query=meta_text,
                top_k_genes=", ".join(
                    tokenized_cell[0].values[: self.config.exp.top_k_genes]
                ),
            )
        else:
            messages_list = self.prompt.format(query=meta_text)
            
        return messages_list, {
            "source_id": source_id,
            "model": self.config.exp.model,
            "temperature": self.config.exp.temperature,
            "top_k_genes": self.config.exp.top_k_genes,
            "dataset": self.config.dataset.h5ad_dir,
            "n_files": self.config.dataset.n_files,
            "n_rows_per_file": self.config.dataset.n_rows_per_file,
            "row_idx": row_idx,
            "idx": idx,
        }


def extract_text_annotation(config: DictConfig):
    all_start = time.time()

    utils.preprocess_config(config)
    utils.setup_weave(config)
    # utils.setup_wandb(config)

    dataset, metadata, prompt, llm, model = get_components(config)

    ouptut_file = open(config.exp.output_path, 'a')
    
    # Create dataset and dataloader
    annotation_dataset = AnnotationDataset(dataset, metadata, model, prompt, config)
    dataloader = DataLoader(
        annotation_dataset,
        batch_size=config.exp.batch_size,
        num_workers=4,  # Adjust based on your CPU cores
    )

    all_annotations = []
    all_metadata = []
    
    for batch_messages, batch_metadata in tqdm(dataloader):
        start = time.time()
        annotations = llm.batch(batch_messages)
        end = time.time()
        print(f"LLM call took: {(end - start) / 1000} seconds")

        all_annotations.extend(annotations)
        for i in range(config.exp.batch_size):
            current_meta = {}
            for k in batch_metadata.keys():
                current_meta[k] = batch_metadata[k][i]
            all_metadata.append(current_meta)

        for annotation, batch_metadata_item in zip(all_annotations, all_metadata):
            ouptut_file.write(f"{batch_metadata_item['source_id']},{batch_metadata_item['row_idx']},{annotation}\n")
            ouptut_file.flush()

    ouptut_file.close()
    all_end = time.time()

    print(f"ALL took: {(all_end - all_start) / 1000} seconds")

    
