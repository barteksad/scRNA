import dotenv
import hydra
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf
import torch

import warnings

warnings.simplefilter("ignore")

OmegaConf.register_resolver("torch_dtype", lambda name: getattr(torch, name))


@hydra.main(version_base=None, config_path="../configs")
def main(config: DictConfig):
    call(config.exp.run_func, config)


if __name__ == "__main__":
    dotenv.load_dotenv(override=True)
    main()
