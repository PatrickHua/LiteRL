"""
Trainer - Main training logic.
Receives preprocessed samples from sampler and performs RL training.
"""

import logging
import torch
from pathlib import Path
from queue import Queue
from typing import Dict, Any
from omegaconf import DictConfig

from .training_utils import (
    load_model_and_tokenizer,
    tokenize_sample,
    collate_batch,
    rl_step,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer that receives samples from queue, accumulates batches, and performs RL training."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.samples_received = 0
        self.samples_logged = 0
        self.log_every_n = 5  # Log every Nth sample for inspection
        
        # Training configuration
        train_cfg = cfg.training
        self.batch_size = train_cfg.batch_size
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
        self.max_seq_length = train_cfg.max_seq_length
        self.save_every_n_steps = train_cfg.save_every_n_steps
        
        # Device setup
        device_str = train_cfg.device
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        self.model, self.tokenizer = load_model_and_tokenizer(cfg)
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        learning_rate = train_cfg.learning_rate
        weight_decay = train_cfg.weight_decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        logger.info(f"Initialized optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
        
        # Batch accumulation
        self.batch_buffer = []
        self.training_step = 0
        self.gradient_accumulation_counter = 0  # Track gradient accumulation
        
        # Output directory
        self.output_dir = Path(cfg.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self, sample_queue: Queue):
        """Run trainer loop, reading samples from queue and training on batches.
        
        Args:
            sample_queue: Queue containing preprocessed samples from sampler
        """
        logger.info("Starting trainer...")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"Max sequence length: {self.max_seq_length}")
        logger.info(f"Will log every {self.log_every_n} samples for inspection")
        logger.info("Trainer is now waiting for samples from sampler...")
        
        # Get rollout config for tokenization
        system_prompt = self.cfg.rollout.get("system_prompt", "")
        task_template = self.cfg.rollout.get("task_template", "{task}")
        
        while True:
            # Get sample from queue (blocks until available)
            # This will receive samples as soon as they're put by the sampler
            sample = sample_queue.get()
            
            # Check for sentinel value (None) indicating end of sampling
            if sample is None:
                logger.info(f"Received end signal. Total samples received: {self.samples_received}")
                # Process any remaining samples in buffer
                if self.batch_buffer:
                    logger.info(f"Processing final batch with {len(self.batch_buffer)} samples")
                    self._process_batch(system_prompt, task_template)
                break
            
            self.samples_received += 1
            
            # Log when sample is received (shows concurrent behavior)
            logger.info(f"[Trainer] Received sample #{self.samples_received} (problem ID: {sample.get('problem', {}).get('id', 'unknown')}, reward: {sample.get('reward', 0.0):.4f})")
            
            # Log sample for inspection periodically
            if self.samples_received % self.log_every_n == 0:
                self.log_sample(sample, self.samples_received)
            
            # Accumulate sample in batch buffer
            self.batch_buffer.append(sample)
            
            # Process batch when buffer is full
            if len(self.batch_buffer) >= self.batch_size:
                self._process_batch(system_prompt, task_template)
        
        logger.info(f"Trainer finished. Processed {self.samples_received} samples total")
        logger.info(f"Completed {self.training_step} training steps")
    
    def _process_batch(self, system_prompt: str, task_template: str):
        """Process accumulated batch: tokenize, collate, train, and save checkpoint if needed.
        
        Args:
            system_prompt: System prompt for tokenization
            task_template: Task template for tokenization
        """
        if not self.batch_buffer:
            return
        
        logger.info(f"Processing batch of {len(self.batch_buffer)} samples (training step {self.training_step + 1})")
        
        try:
            # Tokenize all samples in batch
            tokenized_samples = []
            for sample in self.batch_buffer:
                try:
                    tokenized = tokenize_sample(
                        sample,
                        self.tokenizer,
                        self.max_seq_length,
                        system_prompt,
                        task_template,
                    )
                    tokenized_samples.append(tokenized)
                except Exception as e:
                    logger.error(f"Error tokenizing sample: {e}", exc_info=True)
                    continue
            
            if not tokenized_samples:
                logger.warning("No valid tokenized samples in batch, skipping")
                self.batch_buffer.clear()
                return
            
            # Collate into batch
            batch = collate_batch(tokenized_samples, self.tokenizer)
            
            # Move tensors to device (rewards/logprobs are lists, not tensors)
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Perform training step with gradient accumulation
            self.gradient_accumulation_counter += 1
            is_accumulation_step = (self.gradient_accumulation_counter % self.gradient_accumulation_steps != 0)
            
            loss, metrics = rl_step(
                self.model, 
                batch, 
                self.optimizer, 
                self.device,
                accumulate_gradients=is_accumulation_step,
                gradient_accumulation_steps=self.gradient_accumulation_steps
            )
            
            # Only increment training step and update optimizer after accumulation completes
            if not is_accumulation_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.training_step += 1
                
                # Log metrics
                logger.info(f"Training step {self.training_step}: loss={loss:.6f}, mean_reward={metrics['mean_reward']:.4f}, "
                           f"num_tokens={metrics['num_generation_tokens']}")
                
                # Save checkpoint if needed
                if self.training_step % self.save_every_n_steps == 0:
                    save_checkpoint(
                        self.model,
                        self.tokenizer,
                        self.optimizer,
                        self.training_step,
                        self.output_dir,
                    )
            else:
                # Log accumulation progress
                logger.debug(f"Gradient accumulation: {self.gradient_accumulation_counter % self.gradient_accumulation_steps}/{self.gradient_accumulation_steps}")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
        finally:
            # Clear batch buffer
            self.batch_buffer.clear()
    
    def log_sample(self, sample: Dict[str, Any], sample_num: int):
        """Log a sample for inspection.
        
        Args:
            sample: Preprocessed sample dictionary
            sample_num: Sample number (for logging)
        """
        self.samples_logged += 1
        logger.info("=" * 60)
        logger.info(f"Sample #{sample_num} (logged sample #{self.samples_logged}):")
        logger.info(f"  Problem ID: {sample.get('problem', {}).get('id', 'unknown')}")
        logger.info(f"  Reward: {sample.get('reward', 0.0):.4f}")
        logger.info(f"  Finished: {sample.get('finished', False)}")
        logger.info(f"  Answer Status: {sample.get('answer_status', 'unknown')}")
        logger.info(f"  Num Tokens: {len(sample.get('tokens', []))}")
        logger.info(f"  Num Logprobs: {len(sample.get('logprobs', []))}")
        logger.info(f"  Generated Text (first 300 chars):")
        text = sample.get('text', '')
        logger.info(f"    {text[:300]}...")
        logger.info("=" * 60)

