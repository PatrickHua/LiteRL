"""
vLLM Server Wrapper
Manages starting and stopping vLLM server for inference.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional
import requests
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def wait_for_server(url: str, timeout: int = 300, process: Optional[subprocess.Popen] = None, log_file: Optional[Path] = None):
    """Wait for vLLM server to be ready.
    
    Args:
        url: Server URL to check
        timeout: Maximum time to wait in seconds
        process: Optional subprocess to monitor for crashes
        log_file: Optional log file path to read error messages from
    """
    logger.info(f"Waiting for vLLM server at {url} to be ready...")
    start_time = time.time()
    last_log_time = start_time
    
    # Use localhost for client connections if host is 0.0.0.0
    client_url = url.replace("0.0.0.0", "localhost")
    
    while True:
        # Check if process crashed
        if process is not None:
            return_code = process.poll()
            if return_code is not None:
                error_msg = f"vLLM server process crashed with exit code {return_code}"
                if log_file and log_file.exists():
                    try:
                        with open(log_file, "r") as f:
                            lines = f.readlines()
                            last_lines = lines[-20:] if len(lines) > 20 else lines
                            error_msg += f"\nLast 20 lines of log file:\n" + "".join(last_lines)
                    except Exception as e:
                        error_msg += f"\nCould not read log file: {e}"
                raise RuntimeError(error_msg)
        
        # Check server health
        try:
            response = requests.get(f"{client_url}/health", timeout=5)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                logger.info(f"vLLM server at {url} is ready! (took {elapsed:.1f}s)")
                return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            error_msg = f"vLLM server at {url} did not become ready within {timeout}s (elapsed: {elapsed:.1f}s)"
            if log_file and log_file.exists():
                try:
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        last_lines = lines[-20:] if len(lines) > 20 else lines
                        error_msg += f"\nLast 20 lines of log file:\n" + "".join(last_lines)
                except Exception as e:
                    error_msg += f"\nCould not read log file: {e}"
            raise TimeoutError(error_msg)
        
        # Log progress every 10 seconds
        if time.time() - last_log_time >= 10.0:
            elapsed = time.time() - start_time
            logger.info(f"Still waiting for vLLM server... (elapsed: {elapsed:.1f}s)")
            last_log_time = time.time()
        
        time.sleep(3.0)


class VLLMServer:
    """Manages vLLM server process."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.process: Optional[subprocess.Popen] = None
        self.url = f"http://{cfg.vllm.server.host}:{cfg.vllm.server.port}"
        
    def start(self):
        """Start vLLM server as subprocess."""
        model_path = self.cfg.model_path
        port = self.cfg.vllm.server.port
        host = self.cfg.vllm.server.host
        
        # Get GPU device IDs
        gpu_ids = self.cfg.gpu.actor.device_ids
        gpu_str = ",".join([str(gpu) for gpu in gpu_ids])
        
        # Build vLLM command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_path),
            "--host", host,
            "--port", str(port),
            "--dtype", self.cfg.vllm.dtype,
            "--gpu-memory-utilization", str(self.cfg.vllm.gpu_memory_utilization),
            "--max-model-len", str(self.cfg.vllm.max_model_len),
            "--tensor-parallel-size", str(self.cfg.gpu.actor.tensor_parallel_size),
            "--pipeline-parallel-size", str(self.cfg.gpu.actor.pipeline_parallel_size),
            "--max-num-seqs", str(self.cfg.vllm.max_num_seqs),
            "--max-num-batched-tokens", str(self.cfg.vllm.max_num_batched_tokens),
        ]
        
        if self.cfg.vllm.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        if self.cfg.vllm.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.cfg.vllm.get("disable_log_requests", False):
            cmd.append("--disable-log-requests")
        if self.cfg.vllm.get("disable_frontend_multiprocessing", False):
            cmd.append("--disable-frontend-multiprocessing")
        
        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        logger.info(f"Using GPUs: {gpu_str}")
        
        # Set CUDA_VISIBLE_DEVICES
        env = dict(os.environ)
        if self.cfg.gpu.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu.cuda_visible_devices
        else:
            env["CUDA_VISIBLE_DEVICES"] = gpu_str
        
        # Start process
        log_file = Path(self.cfg.paths.output_dir) / "logs" / "vllm.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Store log_file path for monitoring
        self.log_file = log_file
        
        with open(log_file, "a") as log_f:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
        
        logger.info(f"vLLM server started with PID {self.process.pid}")
        
        # Wait for server to be ready (monitor process and log file)
        wait_for_server(self.url, process=self.process, log_file=self.log_file)
        
    def stop(self):
        """Stop vLLM server."""
        if self.process:
            logger.info("Stopping vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM server did not terminate gracefully, killing...")
                self.process.kill()
                self.process.wait()
            logger.info("vLLM server stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

