#!/usr/bin/env python3
"""
LiteRL Inference Entry Point
Main entry point for the sampler process (standalone sampling).
"""

import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path

from core.sampler import Sampler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main entry point for inference/sampling process (standalone).
    
    This will:
    1. Start vLLM server
    2. Load problems
    3. Sample outputs
    4. Preprocess (compute reward, add ref_logprobs)
    
    Note: For orchestrated sampling + training, use orchestrate.py instead.
    """
    # Disable struct mode to allow dynamic access to all config keys
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    logger.info("=" * 60)
    logger.info("Starting LiteRL Inference (Standalone Sampling)")
    logger.info("=" * 60)
    logger.info(f"Model: {cfg.model_path}")
    logger.info(f"Task: {cfg.task_name}")
    logger.info(f"Output dir: {cfg.paths.output_dir}")
    
    # Create output directories
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.streams_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Create and run sampler (without trainer)
    sampler = Sampler(cfg)
    
    # For MVP, sample a small number of problems
    max_problems = cfg.get("max_problems", 10)
    max_concurrent = cfg.vllm.max_concurrent
    logger.info(f"Sampling {max_problems} problems")
    sampler.run(max_problems=max_problems, max_concurrent=max_concurrent, sample_queue=None)
    
    logger.info("Sampling complete!")


if __name__ == "__main__":
    main()

