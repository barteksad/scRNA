import wandb
import weave
from pathlib import Path
import omegaconf
from omegaconf import DictConfig
import os

def extract_output_dir(config: DictConfig) -> Path:
    '''
    Extracts path to output directory created by Hydra as pathlib.Path instance
    '''
    date = '/'.join(list(config._metadata.resolver_cache['now'].values()))
    output_dir = Path.cwd() / 'outputs' / date
    return output_dir

def preprocess_config(config):
    config.exp.log_dir = extract_output_dir(config)

def setup_wandb(config):
    group, name = str(config.exp.log_dir).split('/')[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve = True, throw_on_missing = True
    )
    name = os.getenv("WANDB_RUN_NAME", name)
    
    if "id" in config.wandb.keys() and config.wandb.id is not None:
        run_id = config.wandb.id
    else:
        run_id = None

    return wandb.init(
        project = config.wandb.project,
        entity = config.wandb.entity,
        dir = config.exp.log_dir,
        group = group,
        name = name,
        config = wandb_config,
        sync_tensorboard = True,
        tags = config.wandb.tags,
        id = run_id
    )

def setup_weave(config):
    group, name = str(config.exp.log_dir).split('/')[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve = True, throw_on_missing = True
    )
    name = os.getenv("WANDB_RUN_NAME", name)
    
    if "id" in config.wandb.keys() and config.wandb.id is not None:
        run_id = config.wandb.id
    else:
        run_id = None

    return weave.init(
        project_name = config.wandb.project,
    )