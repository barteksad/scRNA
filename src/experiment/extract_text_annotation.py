from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm
import utils
from langchain_core.prompts import BaseChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models import BaseSingleCellModel
import weave

import logging
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
    
    for i, (x, obs, var) in tqdm(enumerate(dataset)):
        tokenized_cell = model.tokenize_single_cell(x[None, :], obs, var)
        meta_text = metadata.get_metadata(obs, var)
        if "top_k_genes" in prompt.input_variables:
            messages_list = prompt.format(query=meta_text, top_k_genes=", ".join(tokenized_cell[0].values[:config.exp.top_k_genes]))
        else:
            messages_list = prompt.format(query=meta_text)
        # run it as a separate function to see clear trace on weave
        annotations =  generate_text_annotation(messages_list)
        print(annotations)