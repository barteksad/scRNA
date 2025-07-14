#!/usr/bin/env python3

import os
import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from experiment.clip import train_clip


@hydra.main(config_path="../configs", config_name="clip", version_base=None)
def main(config: DictConfig):
    """Main entry point for CLIP training"""
    # Set up distributed training if environment variables are present
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        print(
            f"Detected distributed training environment: "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
            f"RANK={os.environ.get('RANK')}, "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
        )

    # Call the training function
    train_clip(config)


if __name__ == "__main__":
    main()
