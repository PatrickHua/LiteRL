"""
Math problem solving task - rollout generation.
"""

import logging
import time
import aiohttp
from omegaconf import DictConfig
from typing import Dict, Any
from openai import AsyncOpenAI



logger = logging.getLogger(__name__)


async def generate_math_rollout(
    cfg: DictConfig,
    problem: Dict[str, Any],
    session: aiohttp.ClientSession,
    vllm_url: str,
) -> Dict[str, Any]:
    """
    Generate a rollout for a math problem.
    
    Args:
        cfg: Configuration
        problem: Problem dictionary with 'task' and 'answer' fields
        session: aiohttp session for making requests
        vllm_url: URL of vLLM server
        
    Returns:
        Dictionary with 'text', 'logprobs', 'tokens', 'reward', etc.
    """
    # Build messages
    messages = []
    if cfg.rollout.system_prompt:
        messages.append({"role": "system", "content": cfg.rollout.system_prompt})
    task_text = cfg.rollout.task_template.format(task=problem["task"])
    messages.append({"role": "user", "content": task_text})
    # breakpoint()
    # Call vLLM API
    time_start = time.time()
    # breakpoint()
    # client = OpenAI(base_url=f"{vllm_url}/v1", api_key="token-abc123")
    client = AsyncOpenAI(base_url=f"{vllm_url}/v1", api_key="token-abc123")
    response = await client.chat.completions.create(
        model=cfg.model_path,
        messages=messages,
        temperature=cfg.generation.temperature,
        max_tokens=cfg.generation.max_tokens,
        logprobs=True,  # Get logprobs for RL
    )
    
    latency = time.time() - time_start
    
    # Extract generation
    choice = response.choices[0]
    text = choice.message.content
    
    # Extract logprobs if available
    logprobs = []
    tokens = []
    if choice.logprobs and choice.logprobs.content:  # Use .logprobs instead of ["logprobs"]
        token_logprobs = choice.logprobs.content  # Use .content instead of .get("content", [])
        for token_info in token_logprobs:
            if isinstance(token_info, dict):
                logprobs.append(token_info.get("logprob", 0.0))
                tokens.append(token_info.get("token", ""))
            else:
                logprobs.append(0.0)
                tokens.append("")
    finished = choice.finish_reason == "stop"  # or check if it's not "length"
    # For now, return basic structure
    # Reward computation will be added later
    return {
        "text": text,
        "logprobs": logprobs,
        "tokens": tokens,
        "problem": problem,
        "latency": latency,
        "finished": finished,
    }


if __name__ == "__main__":
    import asyncio
    import sys
    import os
    from pathlib import Path
    
    # Fix import path: when running as module, need parent directory in path
    # __file__ is literl/tasks/math/rollout.py
    # We need PipelineRL directory in path so 'literl' can be imported
    current_file = Path(__file__).resolve()
    # Go up from rollout.py -> math -> tasks -> literl -> PipelineRL
    pipeline_dir = current_file.parent.parent.parent.parent
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from literl.core.vllm_server import VLLMServer
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    print("=" * 60)
    print("Testing Math Rollout Generation")
    print("=" * 60)
    
    # Load config
    try:
        # Try to load from config files
        # Hydra requires relative path, so use relative path from current file
        # From literl/tasks/math/rollout.py -> ../../configs
        config_path = "../../configs"
        hydra.initialize(config_path=config_path, version_base="1.3")
        cfg = hydra.compose(config_name="config")
        print("✓ Loaded config from files")
    except Exception as e:
        print(f"Warning: Could not load config files: {e}")
        print("Creating minimal config...")
        # Create minimal config for testing
        cfg_dict = {
            "model": {
                "model_path": "Qwen/Qwen3-0.6B",
                "generation": {
                    "temperature": 1.0,
                    "max_tokens": 512,
                }
            },
            "infra": {
                "server": {
                    "host": "0.0.0.0",
                    "vllm_port": 8080,
                },
                "gpu": {
                    "actor": {
                        "device_ids": [0],
                        "tensor_parallel_size": 1,
                        "pipeline_parallel_size": 1,
                    },
                    "cuda_visible_devices": None,
                },
                "paths": {
                    "output_dir": "outputs",
                }
            },
            "task": {
                "rollout": {
                    "system_prompt": "Please reason step by step, and put your final answer within \\boxed{}.",
                    "task_template": "{task}",
                }
            }
        }
        cfg = OmegaConf.create(cfg_dict)
        # Set struct mode to False to allow accessing nested keys
        OmegaConf.set_struct(cfg, False)
    
    # Test problem
    test_problem = {
        "id": 0,
        "dataset": "test",
        "task": "What is 2 + 2?",
        "answer": "\\boxed{4}",
    }
    
    async def test_rollout():
        """Test rollout generation."""
        vllm_server = None
        session = None
        
        try:
            # Check if vLLM server is already running
            vllm_url = f"http://{cfg.vllm.server.host}:{cfg.vllm.server.port}"
            import requests
            try:
                response = requests.get(f"{vllm_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"\n[1] Using existing vLLM server at {vllm_url}")
                else:
                    raise requests.exceptions.ConnectionError()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                # Start vLLM server
                print(f"\n[1] Starting vLLM server...")
                vllm_server = VLLMServer(cfg)
                vllm_server.start()
                print(f"✓ vLLM server started at {vllm_url}")
            
            # Create HTTP session
            print(f"\n[2] Creating HTTP session...")
            session = aiohttp.ClientSession()
            
            # Generate rollout
            print(f"\n[3] Generating rollout for test problem...")
            print(f"  Problem: {test_problem['task']}")
            print(f"  Expected answer: {test_problem['answer']}")
            
            rollout = await generate_math_rollout(
                cfg=cfg,
                problem=test_problem,
                session=session,
                vllm_url=vllm_url,
            )
            
            # Print results
            print(f"\n[4] Rollout Results:")
            print(f"  Generated text (first 500 chars):")
            print(f"    {rollout['text'][:500]}...")
            print(f"  Latency: {rollout['latency']:.2f}s")
            print(f"  Number of tokens: {len(rollout['tokens'])}")
            print(f"  Number of logprobs: {len(rollout['logprobs'])}")
            if rollout['logprobs']:
                avg_logprob = sum(rollout['logprobs']) / len(rollout['logprobs'])
                print(f"  Average logprob: {avg_logprob:.4f}")
            print(f"  Finished: {rollout['finished']}")
            
            print("\n" + "=" * 60)
            print("✓ Rollout test completed successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n✗ Error during rollout test: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            # Cleanup
            if session:
                await session.close()
            if vllm_server:
                vllm_server.stop()
                print("\n✓ vLLM server stopped")
    
    # Run async test
    asyncio.run(test_rollout())
