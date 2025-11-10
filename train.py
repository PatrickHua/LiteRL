#!/usr/bin/env python3
"""
LiteRL Training Entry Point
Main entry point for the trainer process.
"""

import hydra
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main entry point for training process.
    
    This will:
    1. Read training batches from stream
    2. Train model
    3. Save checkpoints
    4. Signal weight updates
    """
    logger.info("Starting LiteRL Training")
    logger.info(f"Config: {cfg}")
    
    # TODO: Will be implemented later
    logger.info("Trainer not implemented yet. Config loaded successfully!")


if __name__ == "__main__":
    main()

