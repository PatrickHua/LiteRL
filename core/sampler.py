"""
Sampler - Main sampling logic.
Loads problems, samples from vLLM, and prepares data for training.
"""

import asyncio
import logging
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig
import aiohttp
from queue import Queue

from .vllm_server import VLLMServer

logger = logging.getLogger(__name__)


class Sampler:
    """Main sampler that orchestrates sampling from vLLM."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.vllm_server: VLLMServer = None
        self.problems: List[Dict] = []
        self.session: aiohttp.ClientSession = None
        
    def load_problems(self):
        """Load problems from dataset."""
        logger.info("Loading problems from dataset...")
        
        # Import and call the loader function with kwargs

        
        func_config = self.cfg.train_loader.function
        func_path = func_config.name
        kwargs = {k: v for k, v in func_config.items() if k != "name"}
        

        
        # Split module path and function name
        module_path, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        load_func = getattr(module, func_name)
        
        # Call function with kwargs
        self.problems = load_func(**kwargs)
        logger.info(f"Loaded {len(self.problems)} problems")
        
    def start_vllm_server(self):
        """Start vLLM server."""
        logger.info("Starting vLLM server...")
        self.vllm_server = VLLMServer(self.cfg)
        self.vllm_server.start()
        logger.info(f"vLLM server ready at {self.vllm_server.url}")
        
    async def sample_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Sample a single problem."""
        # Import rollout function dynamically
        rollout_path = self.cfg.rollout.function
        module_path, func_name = rollout_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        rollout_func = getattr(module, func_name)
        
        # Generate rollout
        rollout = await rollout_func(
            cfg=self.cfg,
            problem=problem,
            session=self.session,
            vllm_url=self.vllm_server.url,
        )
        
        return rollout
    
    def preprocess_rollout(self, rollout: Dict[str, Any], problem: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess rollout: compute reward and prepare RL fields.
        
        Args:
            rollout: Rollout dictionary with 'text', 'logprobs', 'tokens', 'finished'
            problem: Problem dictionary with 'answer' field
            
        Returns:
            Preprocessed rollout with 'reward' and 'ref_logprobs' added
        """
        # Import reward function dynamically
        reward_path = self.cfg.reward.function
        module_path, func_name = reward_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        reward_func = getattr(module, func_name)
        
        # Compute reward
        reward_config = self.cfg.reward.get("config", None)
        discount_factor = self.cfg.reward.get("discount_factor", 1.0)
        max_tokens = self.cfg.generation.max_tokens
        
        reward = reward_func(
            rollout=rollout,
            problem=problem,
            reward_config=reward_config,
            discount_factor=discount_factor,
            max_tokens=max_tokens,
        )
        
        # Add reward to rollout
        rollout["reward"] = reward
        
        # Set ref_logprobs = logprobs (no reference model for MVP)
        rollout["ref_logprobs"] = rollout.get("logprobs", [])
        
        return rollout
    
    async def run_loop(self, max_problems: int = None, max_concurrent: int = None, sample_queue: Optional[Queue] = None):
        """Main sampling loop with concurrent processing.
        
        Args:
            max_problems: Maximum number of problems to sample
            max_concurrent: Maximum number of concurrent requests to vLLM server.
                          If None, defaults to cfg.vllm.max_num_seqs
            sample_queue: Optional queue to stream preprocessed samples to trainer
        """

        logger.info("Starting sampling loop...")
        
        # Create HTTP session with proper timeout configuration
        # Match the configuration from pipelinerl/actor.py which works reliably
        connector = aiohttp.TCPConnector(limit=50000, limit_per_host=50000, keepalive_timeout=1.0)
        timeout = aiohttp.ClientTimeout(total=3600.0, connect=3600.0, sock_read=3600.0)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
        try:
            # Sample problems
            problems_to_sample = self.problems[:max_problems] if max_problems else self.problems
            logger.info(f"Sampling {len(problems_to_sample)} problems with max_concurrent={max_concurrent}")
            
            # Semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def sample_with_semaphore(problem, index):
                """Sample a problem with semaphore control and preprocessing."""
                async with semaphore:
                    logger.info(f"Starting problem {index+1}/{len(problems_to_sample)}: {problem.get('id', 'unknown')}")
                    try:
                        # Sample from vLLM
                        rollout = await self.sample_problem(problem)
                        
                        # Preprocess immediately after sampling
                        preprocessed = self.preprocess_rollout(rollout, problem)
                        
                        logger.info(f"Completed problem {index+1}/{len(problems_to_sample)}: {problem.get('id', 'unknown')}")
                        logger.info(f"  Generated text (first 200 chars): {preprocessed['text'][:200]}...")
                        logger.info(f"  Latency: {preprocessed['latency']:.2f}s")
                        logger.info(f"  Reward: {preprocessed['reward']:.4f}")
                        logger.info(f"  Logprobs: {len(preprocessed['logprobs'])} tokens")
                        
                        # Stream to trainer queue if provided
                        if sample_queue is not None:
                            sample_queue.put(preprocessed)
                            logger.info(f"  → Sent sample to trainer queue")
                        
                        return preprocessed
                    except Exception as e:
                        logger.error(f"Error sampling problem {index+1}: {e}", exc_info=True)
                        raise
            
            # Create all tasks concurrently
            tasks = [
                sample_with_semaphore(problem, i)
                for i, problem in enumerate(problems_to_sample)
            ]
            
            # Wait for all tasks to complete
            rollouts = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            successful_rollouts = []
            for i, result in enumerate(rollouts):
                if isinstance(result, Exception):
                    logger.error(f"Problem {i+1} failed with exception: {result}")
                else:
                    successful_rollouts.append(result)
            
            logger.info(f"Completed {len(successful_rollouts)}/{len(problems_to_sample)} problems successfully")
            
            # Signal end of sampling by putting None in queue
            if sample_queue is not None:
                sample_queue.put(None)  # Sentinel value to signal completion
            
        finally:
            await self.session.close()
    
    def run(self, max_problems: int = None, max_concurrent: int = None, sample_queue: Optional[Queue] = None):
        """Run sampler (synchronous entry point).
        
        Args:
            max_problems: Maximum number of problems to sample
            max_concurrent: Maximum number of concurrent requests to vLLM server.
                          If None, defaults to cfg.vllm.max_num_seqs
            sample_queue: Optional queue to stream preprocessed samples to trainer
        """

        
        # Load problems
        self.load_problems()
        
        # Start vLLM server
        self.start_vllm_server()
        
        try:
            # Run async loop
            asyncio.run(self.run_loop(max_problems=max_problems, max_concurrent=max_concurrent, sample_queue=sample_queue))
        finally:
            # Stop vLLM server
            if self.vllm_server:
                self.vllm_server.stop()


if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    
    # Fix import path
    current_file = Path(__file__).resolve()  # Get actual file location
    core_dir = current_file.parent  # core/ directory
    literl_dir = core_dir.parent  # literl/ directory
    pipeline_dir = literl_dir.parent  # PipelineRL directory
    
    if str(literl_dir) not in sys.path:
        sys.path.insert(0, str(literl_dir))
    
    import hydra
    from omegaconf import OmegaConf
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    print("=" * 60)
    print("Testing Sampler")
    print("=" * 60)
    print(f"File location: {current_file}")
    print(f"Literl directory: {literl_dir}")
    print(f"Config directory: {literl_dir / 'configs'}")
    
    # Load config
    try:
        # Verify config directory exists
        config_dir = literl_dir / "configs"
        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
        # Hydra resolves config_path relative to the calling file's directory
        # Since this file is in core/, we need to go up one level: "../configs"
        config_path = "../configs"
        hydra.initialize(config_path=config_path, version_base="1.3")
        cfg = hydra.compose(config_name="config")
        
        # Disable struct mode to allow dynamic access
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg, False)
        
        print("✓ Loaded config from files")
    except Exception as e:
        print(f"Warning: Could not load config files: {e}")
        raise e
    
    try:
        print("\n[1] Creating sampler...")
        sampler = Sampler(cfg)
        print("✓ Sampler created")
        
        print("\n[2] Testing sampler with 3 problems...")
        sampler.run(max_problems=3)
        
        print("\n" + "=" * 60)
        print("✓ Sampler test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during sampler test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

