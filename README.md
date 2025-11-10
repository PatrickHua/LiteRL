# LiteRL

A simplified reinforcement learning framework for training LLMs, inspired by PipelineRL.

## Architecture

LiteRL consists of two main components:

1. **`inference.py`** - Sampler process that:
   - Hosts vLLM server
   - Reads problems
   - Samples outputs
   - Preprocesses (tokenize + compute advantages)
   - Writes to training stream

2. **`train.py`** - Trainer process that:
   - Reads training batches from stream
   - Trains model
   - Saves checkpoints
   - Signals weight updates

## Quick Start

### Installation

**Using uv (recommended - isolated environment):**
```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### Running

**Make sure to activate the virtual environment first:**
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Option 1: Use the launcher (recommended)**
```bash
# Launches both inference and train processes
python launch.py
```

**Option 2: Run separately**
```bash
# Start sampler (in one terminal)
python inference.py

# Start trainer (in another terminal)
python train.py
```

**Option 3: Test sampler directly**
```bash
# Test the sampler with 3 problems
python -m core.sampler
```

## Configuration

Configs are organized in `configs/`:
- `infra.yaml` - Infrastructure (GPU, ports, paths)
- `model.yaml` - Model configuration
- `task.yaml` - Task-specific settings

## Status

🚧 **Work in Progress** - MVP for math problem solving task.

