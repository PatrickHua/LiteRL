#!/usr/bin/env python3
"""
LiteRL Orchestration
Orchestrates sampler and trainer to show their interaction dynamics.

This file demonstrates the MVP architecture:
- Sampler: samples from vLLM → preprocesses → streams to queue
- Trainer: receives from queue → accumulates → (TBD: performs training)
"""

import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import threading
from queue import Queue

from core.sampler import Sampler
from core.trainer import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main orchestration entry point.
    
    This demonstrates the dynamics between sampler and trainer:
    1. Creates a queue for communication
    2. Starts trainer in a separate thread (consumer)
    3. Runs sampler in main thread (producer)
    4. Sampler streams preprocessed samples to trainer via queue
    5. Trainer receives samples and logs them for inspection
    """
    # Disable struct mode to allow dynamic access to all config keys
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    
    logger.info("=" * 60)
    logger.info("LiteRL Orchestration")
    logger.info("=" * 60)
    logger.info(f"Model: {cfg.model_path}")
    logger.info(f"Task: {cfg.task_name}")
    logger.info(f"Output dir: {cfg.paths.output_dir}")
    logger.info("")
    logger.info("Architecture:")
    logger.info("  Sampler (producer) → Queue → Trainer (consumer)")
    logger.info("=" * 60)
    
    # Create output directories
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.streams_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.logs_dir).mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # Create communication channel between sampler and trainer
    # ============================================================
    sample_queue = Queue()
    logger.info("Created sample queue for sampler → trainer communication")
    
    # ============================================================
    # Initialize Trainer (consumer)
    # ============================================================
    trainer = Trainer(cfg)
    logger.info("Initialized trainer")
    
    # Start trainer in a separate thread
    # Trainer will block on queue.get() until samples arrive
    trainer_thread = threading.Thread(
        target=trainer.run,
        args=(sample_queue,),
        daemon=False,
        name="TrainerThread"
    )
    trainer_thread.start()
    logger.info("Started trainer thread (waiting for samples...)")
    logger.info("")
    
    # ============================================================
    # Initialize and Run Sampler (producer)
    # ============================================================
    sampler = Sampler(cfg)
    logger.info("Initialized sampler")
    
    # Configure sampling
    max_problems = cfg.get("max_problems", 10)
    max_concurrent = cfg.vllm.max_concurrent
    logger.info(f"Sampling {max_problems} problems with max_concurrent={max_concurrent}")
    logger.info("")
    
    # Run sampler - it will:
    # 1. Load problems
    # 2. Start vLLM server
    # 3. Sample each problem concurrently
    # 4. Preprocess each sample (compute reward, add ref_logprobs)
    # 5. Put preprocessed sample into queue (trainer receives it)
    # 6. Put None into queue when done (signals trainer to exit)
    logger.info("=" * 60)
    logger.info("Starting sampling...")
    logger.info("=" * 60)
    sampler.run(
        max_problems=max_problems,
        max_concurrent=max_concurrent,
        sample_queue=sample_queue
    )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Sampling complete. Waiting for trainer to finish processing...")
    logger.info("=" * 60)
    
    # ============================================================
    # Wait for trainer to finish
    # ============================================================
    # Trainer will exit when it receives None from the queue
    trainer_thread.join()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Orchestration complete!")
    logger.info("=" * 60)
    logger.info(f"Trainer processed {trainer.samples_received} samples total")
    logger.info(f"Trainer logged {trainer.samples_logged} samples for inspection")


if __name__ == "__main__":
    main()

