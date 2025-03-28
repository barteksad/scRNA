import logging

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


def extract_text_annotation(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_weave(config)
    # utils.setup_wandb(config)

    dataset, metadata, prompt, llm, model = get_components(config)

    @weave.op()
    def generate_text_annotation(messages_list):
        return llm.invoke(messages_list)

    for i, (x, obs, var, source_id) in tqdm(enumerate(dataset)):
        with weave.attributes(
            {
                "source_id": source_id,
                "model": config.exp.model,
                "temperature": config.exp.temperature,
                "top_k_genes": config.exp.top_k_genes,
                "dataset": config.dataset.h5ad_dir,
                "n_files": config.dataset.n_files,
                "n_rows_per_file": config.dataset.n_rows_per_file,
                "idx": i,
            }
        ):
            if len(x.shape) < 2:
                x = x[None, ...]
            tokenized_cell = model.tokenize_single_cell(x, obs, var)
            meta_text = metadata.get_metadata(obs, var, source_id)
            if "top_k_genes" in prompt.input_variables:
                messages_list = prompt.format(
                    query=meta_text,
                    top_k_genes=", ".join(
                        tokenized_cell[0].values[: config.exp.top_k_genes]
                    ),
                )
            else:
                messages_list = prompt.format(query=meta_text)
            # run it as a separate function to see clear trace on weave
            annotations = generate_text_annotation(messages_list)
            print(annotations)
